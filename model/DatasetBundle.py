from datasets import Dataset

class EtriDataset(Dataset):
    def __init__(
        self,
        file_names, 
        text_embeddings, 
        wav_embeddings, 
        Temp,
        EDA,
        Emotion,
        Emotion_ext, 
        Arousal, 
        Valence
    ):
        
        self.file_names = file_names
        self.text_embeddings = text_embeddings
        self.wav_embeddings = wav_embeddings
        self.temp = Temp
        self.eda = EDA
        self.label_emotion = Emotion
        self.label_emotion_ext = Emotion_ext
        self.label_arousal = Arousal
        self.label_valence = Valence
        
        
    def __len__(
        self
    ):
        
        return len(self.file_names)

    def __getitem__(
        self,
        idx
    ):
    
        text_embeddings = self.text_embeddings[idx]
        wav_embeddings = self.wav_embeddings[idx]
        temp = self.temp[idx]
        eda = self.eda[idx]
        label_emotion = self.label_emotion[idx]
        label_emotion_ext = self.label_emotion_ext[idx]
        label_arousal = self.label_arousal[idx]
        label_valence = self.label_valence[idx]
        
        return text_embeddings, wav_embeddings, temp, eda, label_emotion, label_emotion_ext, label_arousal, label_valence