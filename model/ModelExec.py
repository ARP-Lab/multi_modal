import warnings
warnings.filterwarnings('ignore')

import os

from typing import Union, Tuple
import math
import random

import pandas as pd
import numpy as np

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchsummary import summary as summary

from copy import deepcopy

from collections import Counter

from datasets import Dataset

from torchmetrics import F1Score
from sklearn.metrics import f1_score as f1_skearn
from sklearn.metrics import recall_score as recall_sklearn
from sklearn.metrics import precision_score as precision_sklearn

from prettytable import PrettyTable

from zconf.zconf import zconf
import wandb

from model.DatasetBundle import EtriDataset
from model.MLP import MLP
from model.CNN import (
    CNN_TensorFusionMixer, CNN_TS_First, CNN_TS_Merge
)
from model.Loss import weighted_MSELoss
from model.TensorFusionMixer import TensorFusionMixer


class ModelExec(zconf):
    def __init__(
        self,
        zconf_path: str="",
        zconf_id: str=""
    ):
        
        super().__init__(zconf_path=zconf_path, zconf_path=zconf_path)
        
        self.device = self.glob_conf["device"]
        
        
    def __add_padding(
        self,
        pd_series,
        length: int=50
    ) -> np.ndarray:
        
        _pl = self.local_conf["padding_legnth"] if self.local_conf["padding_legnth"] != None else length
        
        if isinstance(pd_series, float):
            if math.isnan(pd_series):
                return np.zeros(10)
            
        if len(pd_series) < _pl:
            pd_series = np.concatenate([pd_series, np.zeros(_pl - len(pd_series))])        
        elif len(pd_series) == _pl:
            pass
        elif len(pd_series) > _pl:
            pd_series = pd_series[:_pl]
            
        return np.array(pd_series)
        
        
    def __choice_and_remove_list(
        self, 
        original_list,
        k: int=8
    ):
        
        _k = self.local_conf["sample_ramdom_k"] if self.local_conf["sample_ramdom_k"] != None else k
        _rm_new = []
        
        _cl = random.sample(original_list, k=_k)
        for _s in original_list:
            if _s in _cl:
                pass
            else:
                _rm_new.append(_s)
                
        return sorted(_rm_new), sorted(_cl)


    def __session_pick(
        self,
        year: int=None
    ) -> Tuple[list, list, list] :
        
        # session을 train vs test&val로 나눠줌
        _sl = ['Sess0' + str(i + 1) if i < 9 else 'Sess' + str(i + 1) for i in range(40)]
        
        for l in self.local_conf[f"y{year}"]["except"]:
            _sl.remove(l)
            
        _train, _test = self.__choice_and_remove_list(_sl)
        _train, _val = self.__choice_and_remove_list(_train)
        
        return _train, _val, _test
    
    
    def __get_data_by_session(
        self,
        data,
        session_lst
    ) -> Union[pd.DataFrame, dict]:
        
        if isinstance(data, pd.DataFrame):
            for idx, _s in enumerate(session_lst):
                df = pd.DataFrame()
                if idx == 0:
                    df = data[data['Segment ID'].str.startswith(_s)]
                else:
                    df = pd.concat([df, data[data['Segment ID'].str.startswith(_s)]])
                    
            return df
        
        elif isinstance(data, dict):
            _emb_data = {}
            _emb_data['wav'] = []
            _emb_data['txt'] = []
            _emb_data['segment_id'] = []
            
            for _s in session_lst:
                _emb_data['wav'].extend(data[_s]['wav'])
                _emb_data['txt'].extend(data[_s]['txt'])
                _emb_data['segment_id'].extend(data[_s]['segment_id'])
                
            return _emb_data
        
        
    def __reconstruct(
        self,
        emb_data: pd.DataFrame,
        anno_data: pd.DataFrame,
        year: int=None
    ) -> dict:
        
        _emb = {}
        
        for _s in emb_data[0].keys():
            _emb[_s] = {}
            _emb[_s]['wav'] = emb_data[0][_s]
            _emb[_s]['txt'] = emb_data[1][_s]
            _emb[_s]['segment_id'] = anno_data['Segment ID'][anno_data['Segment ID'].str.startswith(_s)]
            
        return _emb
    
    
    def __count_parameters(
        self,
        model
    ):
        table = PrettyTable(["Modules", "Parameters"])
        
        total_params = 0
        
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: 
                continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
            
        print(table)
        print(f"Total Trainable Params: {total_params}")
        
        return total_params
    
    
    def __cal_multiple_class(
        self,
        probs,
        threshold: float=0.083
    ):
        
        _th = self.local_conf["diff_threshold"] if self.local_conf["diff_threshold"] != None else threshold
        
        values = probs.topk(2)
        pred_list = []
        # print(values)
        diffs = abs(torch.diff(values.values))
        
        for idx, diff in zip(values.indices, diffs):
            if diff <= _th:
                sorted_label, idx = torch.sort(idx)
                if sorted_label[0] == 0:
                    pred_list.append(100 * 7 + 10 * sorted_label[1].item())
                else:
                    pred_list.append(100 * sorted_label[0].item() + 10 * sorted_label[1].item())
            else:
                pred_list.append(idx[0].item())
                
        return torch.Tensor(pred_list).to(self.device), probs.argmax(1)
    
    
    def __train(
        self,
        dataloader,
        model,
        loss_fn,
        optimizer
    ):
        
        size = len(dataloader.dataset)    
        # data 순서: text_embeddings, wav_embeddings, temp, eda, label_emotion, label_emotion_ext, label_arousal, label_valence
        for batch, (X_txt, X_wav, X_temp, X_eda, 
                        label_emotion, label_emotion_vec, label_arousal, label_valence) in enumerate(dataloader): 
            y = label_emotion_vec # 라벨을 변경하고자 하면 이 변수만 바꿔주면 나머지는 y로 적용
            # 예측 오류 계산 
            X_txt, X_wav, X_temp, X_eda, y = X_txt.to(self.device), X_wav.to(self.device), X_temp.to(self.device), X_eda.to(self.device),y.type(torch.float32).to(self.device)
            
            X_temp = X_temp.unsqueeze(dim=-1)
            X_eda = X_eda.unsqueeze(dim=-1)
            
            pred = model(X_txt, X_wav, X_temp, X_eda)
            y = F.softmax(y, dim=1)
            
            loss = loss_fn(pred, y)

            # 역전파
            optimizer.zero_grad()
            loss.mean().backward() # weighted MSE를 사용할 경우 중간에 sum() or mean()을 넣어줌
            optimizer.step()
            
            if batch % 100 == 0:
                loss, current = loss.mean().mean().item(), batch * len(X_txt) # weighted MSE를 사용할 경우 중간에 sum() or mean()을 넣어줌
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    
    def __test(
        self,
        dataloader,
        model,
        loss_fn,
        mode='test'
    ):
        
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        
        f1 = F1Score(**self.local_conf["f1_score_conf"]).to(self.device)   
        preds = []
        targets = []
        
        with torch.no_grad():
            # data 순서: text_embeddings, wav_embeddings, temp, eda, label_emotion, label_emotion_ext, label_arousal, label_valence
            for batch, (X_txt, X_wav, X_temp, X_eda, 
                            label_emotion, label_emotion_vec, label_arousal, label_valence) \
                                in enumerate(dataloader):
                y = label_emotion_vec # 라벨을 변경하고자 하면 이 변수만 바꿔주면 나머지는 y로 적용
                
                # 예측 오류 계산
                X_txt, X_wav, X_temp, X_eda, y = \
                    X_txt.to(self.device), X_wav.to(self.device), X_temp.to(self.device), X_eda.to(self.device), y.type(torch.float32).to(self.device)
                
                # shape 변경
                X_temp = X_temp.unsqueeze(dim=-1) # 뒤쪽에 (, 1)
                X_eda = X_eda.unsqueeze(dim=-1) # 뒤쪽에 (, 1)
                
                pred = model(X_txt, X_wav, X_temp, X_eda)
                pred_for_acc, _ = self.__cal_multiple_class(pred)
                
                preds.append(pred_for_acc)
                y = F.softmax(y, dim=1)

                targets.append(label_emotion) # classification을 할 경우 언제나 사용
                # print('예측라벨분포:',pred[:2], '정답라벨 분포:', label_emotion_ext[:2], '예측정답:', pred.argmax(1)[:2],'정답:', label_emotion[:2])
                # print('예측:', pred.argmax(1).tolist()[:2],'\n', '정답:', label_emotion.tolist()[:2])
                # https://discuss.pytorch.org/t/loss-backward-raises-error-grad-can-be-implicitly-created-only-for-scalar-outputs/12152/6
                test_loss += loss_fn(pred, y).mean().item()# weighted MSE를 사용할 경우 중간에 sum() or mean()을 넣어줌 
                
                correct += (pred_for_acc == label_emotion.to(self.device)).type(torch.float).sum().item()
                # correct += (pred.argmax(1) == label_emotion.to(device)).type(torch.float).sum().item()
                
        test_loss /= num_batches
        correct /= size
        # f1_score = f1(torch.cat(preds).to(device), torch.cat(targets).to(device))
        recall_score = \
            recall_sklearn(torch.cat(preds).detach().cpu().numpy(), torch.cat(targets).detach().cpu().numpy(), average="micro")
        recall_score_weighted = \
            recall_sklearn(torch.cat(preds).detach().cpu().numpy(), torch.cat(targets).detach().cpu().numpy(), average="weighted")
        precision_score = \
            precision_sklearn(torch.cat(preds).detach().cpu().numpy(), torch.cat(targets).detach().cpu().numpy(), average="micro")
        precision_score_weighted = \
            precision_sklearn(torch.cat(preds).detach().cpu().numpy(), torch.cat(targets).detach().cpu().numpy(), average="weighted")
        f1_score_weighted = \
            f1_skearn(torch.cat(preds).detach().cpu().numpy(), torch.cat(targets).detach().cpu().numpy(), average="weighted")
        f1_score = \
            f1_skearn(torch.cat(preds).detach().cpu().numpy(), torch.cat(targets).detach().cpu().numpy(), average="micro")
        accuracy = (100 * correct)
        
        if mode == "test":
            print(torch.cat(preds), torch.cat(preds).shape)
            print("f1 score: ", f1_score, "f1 score weighted", f1_score_weighted)
            print(f"Test Error: Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f}\n")
        elif mode == "val":
            print(f"Validation Error: Accuracy: {(accuracy):>0.1f}%, Avg val loss: {test_loss:>8f} \n")
        
        return f1_score, f1_score_weighted, accuracy, test_loss, \
            recall_score, recall_score_weighted, precision_score, precision_score_weighted
    
    
    def proc(
        self,
        emb_data_19: pd.DataFrame,
        emb_data_20: pd.DataFrame,
        anno_data_19: pd.DataFrame,
        anno_data_20: pd.DataFrame,
        save_model: bool=False
    ):
        
        self.act_func = self.local_conf["act_func"] if self.local_conf["act_func"] != None else act_func
        
        encode_dict = self.local_conf["encode_dict"]
        
        encode_dict = {k:v for k, v in sorted(encode_dict.items(), key=lambda item: item[1])}
        decode_dict = {b:i for i, b in encode_dict.items()}
        
        anno_data_19.Emotion = list(anno_data_19.Emotion.map(encode_dict))
        anno_data_20.Emotion = list(anno_data_20.Emotion.map(encode_dict))
        
        anno_data_19['Scaled EDA'] = anno_data_19['Scaled EDA'].apply(self.__add_padding)
        anno_data_20['Scaled EDA'] = anno_data_20['Scaled EDA'].apply(self.__add_padding)
        anno_data_19['Scaled TEMP'] = anno_data_19['Scaled TEMP'].apply(self.__add_padding)
        anno_data_20['Scaled TEMP'] = anno_data_20['Scaled TEMP'].apply(self.__add_padding)
        
        emb_data_19 = self.__reconstruct(emb_data=emb_data_19, anno_data=anno_data_19)
        emb_data_20 = self.__reconstruct(emb_data=emb_data_20, anno_data=anno_data_20)
        
        _train_19, _val_19, _test_19 = self.__session_pick(year=19)
        _train_20, _val_20, _test_20 = self.__session_pick(year=20)
        
        annot_19_train = self.__get_data_by_session(anno_data_19, _train_19)
        annot_20_train = self.__get_data_by_session(anno_data_20, _train_20)
        annot_20_test = self.__get_data_by_session(anno_data_20, _test_20)
        annot_20_val = self.__get_data_by_session(anno_data_20, _val_20)
        
        emb_19_train = self.__get_data_by_session(emb_data_19, _train_19)
        emb_20_train = self.__get_data_by_session(emb_data_20, _train_20)
        emb_20_test = self.__get_data_by_session(emb_data_20, _test_20)
        emb_20_val = self.__get_data_by_session(emb_data_20, _val_20)
        
        target_neutral_num = Counter(kemdy20_annot_train['Emotion'])[4]

        target_neutral_num_19 = 0
        # target_neutral_num_19 = int(target_neutral_num / (Counter(kemdy19_annot['Emotion'])[4] + Counter(kemdy20_annot['Emotion'])[4]) * Counter(kemdy19_annot['Emotion'])[4])
        target_neutral_num_20 = target_neutral_num - target_neutral_num_19
        
        kemdy19_annot_train_not_neut = annot_19_train[annot_19_train['Emotion'] != 4]
        kemdy20_annot_train_not_neut = annot_20_train[annot_20_train['Emotion'] != 4]

        kemdy19_annot_train_neut = annot_19_train[annot_19_train['Emotion'] == 4].sample(target_neutral_num_19)
        kemdy20_annot_train_neut = annot_20_train[annot_20_train['Emotion'] == 4].sample(target_neutral_num_20)
        
        emb_test_final = {}
        emb_test_final['wav'] = []
        emb_test_final['txt'] = []
        emb_test_final['segment_id'] = []

        # emb_test_final['wav'] = kemdy19_emb_test['wav']
        emb_test_final['wav'].extend(emb_20_test['wav'])
        # emb_test_final['txt'] = kemdy19_emb_test['txt']
        emb_test_final['txt'].extend(emb_20_test['txt'])
        # emb_test_final['segment_id'] = kemdy19_emb_test['segment_id']
        emb_test_final['segment_id'].extend(emb_20_test['segment_id'])
        
        emb_val_final = {}
        emb_val_final['wav'] = []
        emb_val_final['txt'] = []
        emb_val_final['segment_id'] = []

        # emb_val_final['wav'] = kemdy19_emb_val['wav']
        emb_val_final['wav'].extend(emb_20_val['wav'])
        # emb_val_final['txt'] = kemdy19_emb_val['txt']
        emb_val_final['txt'].extend(emb_20_val['txt'])
        # emb_val_final['segment_id'] = kemdy19_emb_val['segment_id']
        emb_val_final['segment_id'].extend(emb_20_val['segment_id'])
        
        annot_train_final = pd.concat([kemdy19_annot_train_neut, kemdy20_annot_train_neut, kemdy19_annot_train_not_neut, kemdy20_annot_train_not_neut])
        # annot_test_final = pd.concat([kemdy19_annot_test, kemdy20_annot_test])
        # annot_val_final = pd.concat([kemdy19_annot_val, kemdy20_annot_val])

        annot_test_final = pd.concat([kemdy20_annot_test])
        annot_val_final = pd.concat([kemdy20_annot_val])


        annot_train_final.reset_index(drop=True, inplace=True)
        annot_test_final.reset_index(drop=True, inplace=True)
        annot_val_final.reset_index(drop=True, inplace=True)
        
        # train dataset neutral 4000개로 랜덤 뽑은 것 생성
        emb_train_final = {}
        emb_train_final['wav'] = []
        emb_train_final['txt'] = []
        emb_train_final['segment_id'] = []
        for segment_annot_id in kemdy19_annot_train_neut['Segment ID']:
            for wav, txt, segment_emb_id in zip(emb_19_train['wav'], emb_19_train['txt'], emb_19_train['segment_id']):
                if segment_annot_id == segment_emb_id:
                    emb_train_final['wav'].append(wav)
                    emb_train_final['txt'].append(txt)
                    emb_train_final['segment_id'].append(segment_emb_id)
                    
        for segment_annot_id in kemdy19_annot_train_not_neut['Segment ID']:
            for wav, txt, segment_emb_id in zip(emb_19_train['wav'], emb_19_train['txt'], emb_19_train['segment_id']):
                if segment_annot_id == segment_emb_id:
                    emb_train_final['wav'].append(wav)
                    emb_train_final['txt'].append(txt)
                    emb_train_final['segment_id'].append(segment_emb_id)
                

        for segment_annot_id in kemdy20_annot_train_neut['Segment ID']:
            for wav, txt, segment_emb_id in zip(emb_20_train['wav'], emb_20_train['txt'], emb_20_train['segment_id']):
                if segment_emb_id == segment_annot_id:
                    emb_train_final['wav'].append(wav)
                    emb_train_final['txt'].append(txt)
                    emb_train_final['segment_id'].append(segment_emb_id)
                
        for segment_annot_id in kemdy20_annot_train_not_neut['Segment ID']:
            for wav, txt, segment_emb_id in zip(emb_20_train['wav'], emb_20_train['txt'], emb_20_train['segment_id']):
                if segment_annot_id == segment_emb_id:
                    emb_train_final['wav'].append(wav)
                    emb_train_final['txt'].append(txt)
                    emb_train_final['segment_id'].append(segment_emb_id)
        
        torch.set_default_dtype(torch.float32)
        
        # data load 및 나누기: https://076923.github.io/posts/Python-pytorch-11/

        # annot_train_final, annot_test_final, annot_val_final
        # emb_train_final, emb_test_final, emb_val_final
        # session을 통합시킨 데이터 셋을 만들었을 때
        dataset_train = EtriDataset(file_names=annot_train_final['Segment ID'],
                            text_embeddings=torch.stack(emb_train_final['txt']),
                            wav_embeddings=torch.stack(emb_train_final['wav']),
                            Emotion=annot_train_final['Emotion'],
                            Arousal=annot_train_final['Arousal'],
                            Valence=annot_train_final['Valence'],
                            EDA=torch.Tensor(annot_train_final['Scaled EDA']), 
                            Temp=torch.Tensor(annot_train_final['Scaled TEMP']), 
                            Emotion_vec=torch.Tensor(annot_train_final['emotion_vector'])) 


        dataset_test = EtriDataset(file_names=annot_test_final['Segment ID'],
                            text_embeddings=torch.stack(emb_test_final['txt']),
                            wav_embeddings=torch.stack(emb_test_final['wav']),
                            Emotion=annot_test_final['Emotion'],
                            Arousal=annot_test_final['Arousal'],
                            Valence=annot_test_final['Valence'],
                            EDA=torch.Tensor(annot_test_final['Scaled EDA']), 
                            Temp=torch.Tensor(annot_test_final['Scaled TEMP']), 
                            Emotion_vec=torch.Tensor(annot_test_final['emotion_vector']))

        dataset_val = EtriDataset(file_names=annot_val_final['Segment ID'],
                            text_embeddings=torch.stack(emb_val_final['txt']),
                            wav_embeddings=torch.stack(emb_val_final['wav']),
                            Emotion=annot_val_final['Emotion'],
                            Arousal=annot_val_final['Arousal'],
                            Valence=annot_val_final['Valence'],
                            EDA=torch.Tensor(annot_val_final['Scaled EDA']), 
                            Temp=torch.Tensor(annot_val_final['Scaled TEMP']), 
                            Emotion_vec=torch.Tensor(annot_val_final['emotion_vector']))
        
        train_dataloader = DataLoader(dataset_train, batch_size=512, shuffle=True, drop_last=True)
        validation_dataloader = DataLoader(dataset_val, batch_size=128, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(dataset_test, batch_size=128, shuffle=True, drop_last=True)
        
        # txt_input_length, txt_input_width = raw_dataset[session]['textembeddings'][0].shape | 마지막엔 지울 것
        # , wav_input_length, wav_input_width = raw_dataset[session]['wav_embeddings'][0].shape
        txt_input_length, txt_input_width = torch.Tensor(emb_train_final['txt'][0]).shape
        wav_input_length, wav_input_width = torch.Tensor(emb_train_final['wav'][0]).shape
        # temp_input_length = annot_train_final['Scaled EDA'][0].shape[0]
        # eda_input_length = annot_train_final['Scaled TEMP'][0].shape[0]
        
        # tf_mixer에 들어갈 wav mlp, txt mlp 선언
        model_mlp_txt = MLP(txt_input_length, txt_input_width).to(self.device)
        model_mlp_wav = MLP(wav_input_length, wav_input_width).to(self.device)

        model_cnn_temp = CNN_TS_First().to(self.device)
        model_cnn_eda = CNN_TS_First().to(self.device)

        model_cnn_middle = CNN_TS_Merge().to(self.device)

        model_cnn_final = CNN_TensorFusionMixer().to(self.device)

        # 최종 모델 선언
        model_tf_cnn_mixer = TensorFusionMixer(
            ModelA=model_mlp_txt, 
            ModelB=model_mlp_wav,
            ModelC=model_cnn_temp,
            ModelD=model_cnn_eda,
            ModelE=model_cnn_middle,
            ModelF=model_cnn_final
        ).to(self.device)

        # model 병렬 학습 처리
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model_mlp_txt = nn.DataParallel(model_mlp_txt).to(self.device)
            model_mlp_wav = nn.DataParallel(model_mlp_wav).to(self.device)
            model_cnn_temp = nn.DataParallel(model_cnn_temp).to(self.device)
            model_cnn_eda = nn.DataParallel(model_cnn_eda).to(self.device)
            model_tf_cnn_mixer = nn.DataParallel(model_tf_cnn_mixer).to(self.device)

        # weigted loss for imbalance data: https://naadispeaks.wordpress.com/2021/07/31/handling-imbalanced-classes-with-weighted-loss-in-pytorch/
        # weight 계산
        single_emotion = [_ for _ in range(7)]
        total_obs = 0
        for i in single_emotion:
            total_obs += Counter(annot_train_final['Emotion'])[i]

        print('total (single) obs: ', total_obs)

        weight_for_class = []
        for key, value in sorted(Counter(annot_train_final['Emotion']).items()):
            if key in single_emotion:
                print(f'{key} is in single emotion, {value}')
                weight_for_class.append(1 - (value/total_obs))
                    
        weight_for_class = torch.Tensor(weight_for_class).type(torch.float16)
        
        # loss_fn = nn.CrossEntropyLoss(weight=weight_for_class).to(device)
        # loss_fn = nn.CrossEntropyLoss().to(device) # weigth를 주기위해 위의 loss로 임시 변경
        loss_fn = weighted_MSELoss(weight=weight_for_class).to(self.device) # multi target regression(감정별로 count 한 타겟)
        # loss_fn = nn.MSELoss().to(device)
        
        lr = self.local_conf["lr"]

        optimizer = optim.Adam(model_tf_cnn_mixer.parameters(), lr=lr, weight_decay=0.005)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        
        wandb.login()
        
        #wandb init
        epochs = self.local_conf["epochs"]

        wandb.init(
            # set the wandb project where this run will be logged
            project="ETRI-multiclassification",
            name=f"Experiment Name",
            # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "architecture": "CNN Tensor Fusion Mixer",
                "dataset": "ETRI Kemdy20",
                "epochs": epochs,
                "Optimizer": optimizer.__class__.__name__,
                "Loss": loss_fn.__class__.__name__,
                "multi label threshold": multi_label_threshold,
            }
        )
        
        # Set the Training Parameters
        loss_list = []
        acc_list = []
        recall_list = []
        recall_list_weighted = []
        precision_list = []
        precision_list_weighted = []
        f1_score_list = []
        f1_score_list_weighted = []
        best_acc = 0
        best_f1 = 0
        best_f1_weighted = 0
        best_recall = 0
        best_recall_weighted = 0
        best_precision = 0
        best_precision_weighted = 0

        best_acc_model = None 
        best_f1_model = None
        best_f1_weighted_model = None

        for epoch in range(epochs):
            print(f"---------------Epoch {epoch+1}----------------")
            self.__train(train_dataloader, model_tf_cnn_mixer, loss_fn, optimizer)
            
            f1_score, f1_score_weighted, accuracy, loss,
            recall_score, recall_score_weighted,
            precision_score, precision_score_weighted = self.__test(
                validation_dataloader, model_tf_cnn_mixer, loss_fn, mode='val')
            
            scheduler.step(loss)
            
            if accuracy > best_acc:
                best_acc = accuracy
                best_acc_model = deepcopy(model_tf_cnn_mixer)
                print(f"best_acc: {best_acc:>.2f}%")
            if f1_score > best_f1:
                best_f1 = f1_score
                best_f1_model = deepcopy(model_tf_cnn_mixer)
                print(f"best_f1: {best_f1}")    
            if f1_score_weighted > best_f1_weighted:
                best_f1_weighted = f1_score_weighted
                best_f1_weighted_model = deepcopy(model_tf_cnn_mixer)
                print(f"best_f1_weighted: {best_f1_weighted}")
            if recall_score > best_recall:
                best_recall = recall_score
                # best_recall_model = deepcopy(model_tf_cnn_mixer)
                print(f"best_recall: {recall_score}")
            if recall_score_weighted > best_recall_weighted:
                best_recall_weighted = recall_score_weighted
                # best_recall_weighted_model = deepcopy(model_tf_cnn_mixer)
                print(f"best_recall_weighted: {best_recall_weighted}")
            if precision_score > best_precision:
                best_precision = precision_score
                # best_pre_model = deepcopy(model_tf_cnn_mixer)
                print(f"best_precision: {precision_score}")
            if precision_score_weighted > best_precision_weighted:
                best_precision_weighted = precision_score_weighted
                # best_pre_weighted_model = deepcopy(model_tf_cnn_mixer)
                print(f"best_precision_weighted: {best_precision_weighted}")
            
            recall_list.append(recall_score)
            recall_list_weighted.append(recall_score_weighted)
            precision_list.append(precision_score)
            precision_list_weighted.append(precision_score_weighted)
            f1_score_list.append(f1_score)
            f1_score_list_weighted.append(f1_score_weighted)  
            loss_list.append(loss)
            acc_list.append(accuracy)
            
            wandb.log({'accuracy': accuracy, 'loss': loss, 'f1 score': f1_score})
        
        wandb.finish()
        print("Done!", f"best f1_score: {best_f1}, f1_weighted: {best_f1_weighted} | best accuracy: {best_acc}")
        
        if self.check_var(self.local_conf, "save", True) is not None:
            torch.save(best_f1_model, self.glob_conf["data_path"] + "/saved_model" + f"model_multilabel_{_now_model_audio}.pkl")