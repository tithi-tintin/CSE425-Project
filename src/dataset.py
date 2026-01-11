import os
import librosa
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def extract_mfcc(file_path, n_mfcc=40):
    """
    Extracts MFCC features from an audio file.
    """
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=10.0)
        if len(y) < 22050: 
            y = np.pad(y, (0, 22050 - len(y)))
            
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc, axis=1)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros(n_mfcc)

def load_audio_features(audio_dir):
    """
    Loads only audio features (for Easy Task).
    """
    features = []
    filenames = []
    
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    for file in sorted(os.listdir(audio_dir)):
        if file.endswith(".wav"):
            path = os.path.join(audio_dir, file)
            mfcc = extract_mfcc(path)
            features.append(mfcc)
            filenames.append(file)

    return np.array(features), filenames

def load_hybrid_data(audio_dir, lyrics_dir, n_text_features=50):
    """
    Loads Audio + Lyrics features (for Medium Task).
    """
    audio_features = []
    lyrics_corpus = []
    filenames = []
    
    # Check directories
    if not os.path.exists(lyrics_dir):
        print(f"Warning: Lyrics dir '{lyrics_dir}' not found. Using dummy text.")
        use_dummy = True
    else:
        use_dummy = False

    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    
    for wav_file in audio_files:
        basename = os.path.splitext(wav_file)[0]
        
        # Audio
        wav_path = os.path.join(audio_dir, wav_file)
        audio_features.append(extract_mfcc(wav_path))
        filenames.append(basename)
        
        # Lyrics
        if not use_dummy:
            txt_path = os.path.join(lyrics_dir, basename + ".txt")
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().replace('\n', ' ')
                    lyrics_corpus.append(text)
            else:
                lyrics_corpus.append("") # Empty string if file missing
        else:
            lyrics_corpus.append("dummy text")
    
    # Vectorize Lyrics (TF-IDF)
    if not lyrics_corpus or all(x == "" for x in lyrics_corpus):
        print("Warning: No lyrics found. Using zeros for text features.")
        text_features = np.zeros((len(audio_features), n_text_features))
        vectorizer = None
    else:
        vectorizer = TfidfVectorizer(max_features=n_text_features, stop_words='english')
        text_features = vectorizer.fit_transform(lyrics_corpus).toarray()
    
    # Concatenate
    X_audio = np.array(audio_features)
    
    # Ensure dimensions match
    if text_features.shape[0] != X_audio.shape[0]:
        # Padding just in case
        diff = X_audio.shape[0] - text_features.shape[0]
        text_features = np.vstack([text_features, np.zeros((diff, n_text_features))])

    X_hybrid = np.hstack((X_audio, text_features))
    
    return X_hybrid, filenames, vectorizer

def load_hybrid_data_with_labels(audio_dir, lyrics_dir, n_text_features=50):
    """
    Loads Audio + Lyrics + Singer Labels (for Hard Task / ARI score).
    Extracts singer ID from filename (e.g., 'amy_1_01' -> 'amy').
    """
    X_hybrid, filenames, vectorizer = load_hybrid_data(audio_dir, lyrics_dir, n_text_features)
    
    singer_labels = []
    for f in filenames:
    
        label = f.split('_')[0]
        singer_labels.append(label)
        
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(singer_labels)
    
    return X_hybrid, y_encoded, le, filenames
