import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import glob
from pathlib import Path
from PIL import Image
import pretty_midi
import tempfile
import io
import time
import music21
from music21 import converter, corpus, instrument, midi, note, chord, pitch, stream
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🎼 Classical Composer Identifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .composer-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Constants
TARGET_COMPOSERS = ['Bach', 'Beethoven', 'Chopin', 'Mozart']
DATA_DIR = Path("data")
MODEL_DIR = DATA_DIR / "model"
FEATURES_DIR = DATA_DIR / "features"
IMAGES_DIR = DATA_DIR / "images"
MIDI_DIR = DATA_DIR / "midiclassics"

# Wikipedia info for composers
COMPOSER_INFO = {
    'Bach': """
    **Johann Sebastian Bach (1685-1750)** 🎼
    
    German composer and musician of the late Baroque period. Known for his complex 
    polyphonic compositions, Bach created timeless masterpieces including the 
    Brandenburg Concertos, The Well-Tempered Clavier, and Mass in B minor. His works 
    demonstrate mathematical precision combined with profound emotional depth.
    
    **Style:** Complex counterpoint, intricate fugues, mathematical structures
    """,
    
    'Beethoven': """
    **Ludwig van Beethoven (1770-1827)** 🎵
    
    German composer who bridged the Classical and Romantic eras. Despite progressive 
    hearing loss, Beethoven composed nine symphonies, 32 piano sonatas, and numerous 
    chamber works. His music is characterized by dramatic contrasts, innovative 
    structures, and emotional intensity.
    
    **Style:** Dramatic dynamics, innovative forms, emotional expression
    """,
    
    'Chopin': """
    **Frédéric Chopin (1810-1849)** 🎹
    
    Polish composer and virtuoso pianist of the Romantic era. Chopin wrote primarily 
    for solo piano, creating works of extraordinary technical brilliance and poetic 
    beauty. His compositions include ballades, nocturnes, études, and polonaises that 
    showcase the piano's expressive capabilities.
    
    **Style:** Lyrical melodies, sophisticated harmonies, pianistic virtuosity
    """,
    
    'Mozart': """
    **Wolfgang Amadeus Mozart (1756-1791)** 🎶
    
    Austrian composer of the Classical period, renowned for his prodigious talent and 
    prolific output. Mozart composed over 600 works including symphonies, operas, 
    chamber music, and concertos. His music exemplifies clarity, balance, and 
    effortless melodic beauty.
    
    **Style:** Elegant melodies, perfect formal balance, classical clarity
    """
}

@st.cache_data
def load_all_app_data():
    """Load all required data for the application: model, features, and MIDI files"""
    with st.spinner("🎵 Loading AI model and features..."):
        # Load model
        model_path = MODEL_DIR / "composer_classification_model_best.keras"
        if not model_path.exists():
            st.error(f"Model file not found at {model_path}")
            return None, None, None
            
        model = keras.models.load_model(model_path)
        
        # Load exported features
        try:
            musical_df = pd.read_pickle(FEATURES_DIR / "musical_features_df.pkl")
            harmonic_df = pd.read_pickle(FEATURES_DIR / "harmonic_features_df.pkl")
            note_sequences = np.load(FEATURES_DIR / "note_sequences.npy")
            sequence_labels = np.load(FEATURES_DIR / "sequence_labels.npy")
            
            # Load note mapping if available
            try:
                with open(FEATURES_DIR / "note_mapping.pkl", 'rb') as f:
                    note_mapping = pickle.load(f)
            except:
                note_mapping = None
            
            exported_features = {
                'musical_df': musical_df,
                'harmonic_df': harmonic_df,
                'note_sequences': note_sequences,
                'sequence_labels': sequence_labels,
                'note_mapping': note_mapping
            }
        except Exception as e:
            st.error(f"Error loading exported features: {str(e)}")
            return None, None, None
        
        # Load MIDI files
        midi_files = {}
        for composer in TARGET_COMPOSERS:
            composer_dir = MIDI_DIR / composer
            if composer_dir.exists():
                midi_files[composer] = [f.name for f in composer_dir.glob("*.mid")]
            else:
                midi_files[composer] = []
        
        return model, exported_features, midi_files

@st.cache_data
def load_model_and_features():
    """Load the trained model"""
    try:
        # Load model
        model_path = MODEL_DIR / "composer_classification_model_best.keras"
        if not model_path.exists():
            st.error(f"Model file not found at {model_path}")
            return None, None
            
        model = keras.models.load_model(model_path)
        
        # Check what data files are available
        files_exist = {}
        loaded_data = {}
        
        # Try to load feature files (for informational purposes)
        musical_features_path = FEATURES_DIR / "musical_features_df.pkl"
        if musical_features_path.exists():
            files_exist['musical'] = True
            try:
                with open(musical_features_path, 'rb') as f:
                    loaded_data['musical'] = pickle.load(f)
            except:
                files_exist['musical'] = False
        else:
            files_exist['musical'] = False
        
        harmonic_features_path = FEATURES_DIR / "harmonic_features_df.pkl"
        if harmonic_features_path.exists():
            files_exist['harmonic'] = True
            try:
                with open(harmonic_features_path, 'rb') as f:
                    loaded_data['harmonic'] = pickle.load(f)
            except:
                files_exist['harmonic'] = False
        else:
            files_exist['harmonic'] = False
        
        return model, {'files_exist': files_exist, 'data': loaded_data}
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

@st.cache_data
def get_midi_files():
    """Get list of MIDI files organized by composer"""
    midi_files = {}
    
    for composer in TARGET_COMPOSERS:
        composer_dir = MIDI_DIR / composer
        if composer_dir.exists():
            files = list(composer_dir.glob("*.mid")) + list(composer_dir.glob("*.midi"))
            midi_files[composer] = [f.name for f in files]
        else:
            midi_files[composer] = []
    
    return midi_files

def load_composer_image(composer):
    """Load composer image"""
    try:
        image_path = IMAGES_DIR / f"{composer}.jpg"
        if image_path.exists():
            return Image.open(image_path)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading image for {composer}: {str(e)}")
        return None

def convert_prettymidi_to_audio(pm, audio_path, duration_limit):
    """
    Fallback function to convert MIDI using pretty_midi library
    """
    try:
        # Synthesize audio using pretty_midi
        audio = pm.synthesize(fs=22050)
        
        # Trim to duration limit
        max_samples = int(duration_limit * 22050)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Save as WAV file
        sf.write(str(audio_path), audio, 22050)
        return str(audio_path)
        
    except Exception as e:
        st.warning(f"Pretty_midi fallback also failed: {str(e)}")
        return None

@st.cache_data
def convert_midi_to_audio(midi_file_path, duration_limit=30):
    """
    Convert MIDI file to audio using music21 and return audio data
    Limited to first 30 seconds for preview
    """
    try:
        # Check if MIDI file exists and is readable
        if not Path(midi_file_path).exists():
            st.error(f"MIDI file not found: {midi_file_path}")
            return None
            
        if Path(midi_file_path).stat().st_size == 0:
            st.warning("MIDI file is empty")
            return None
        
        # Create a temporary directory for audio files
        temp_dir = Path(tempfile.gettempdir()) / "midi_audio_cache"
        temp_dir.mkdir(exist_ok=True)
        
        # Create a unique filename for the cached audio
        audio_filename = f"{Path(midi_file_path).stem}_{duration_limit}s.wav"
        audio_path = temp_dir / audio_filename
        
        # Check if audio file already exists
        if audio_path.exists():
            return str(audio_path)
        
        # Load MIDI file using music21
        try:
            score = converter.parse(str(midi_file_path))
        except Exception as parse_error:
            # If music21 can't parse it, try with pretty_midi as fallback
            try:
                import pretty_midi
                pm = pretty_midi.PrettyMIDI(str(midi_file_path))
                return convert_prettymidi_to_audio(pm, audio_path, duration_limit)
            except ImportError:
                st.warning("Advanced MIDI parsing requires pretty_midi library. Using basic fallback.")
                return None
            except Exception as fallback_error:
                st.warning(f"Both music21 and pretty_midi failed to parse MIDI file. Original error: {str(parse_error)[:100]}")
                return None
        
        # Simplify the score for better audio generation
        simplified_score = score.flatten()
        
        # Get the notes and create a simple sine wave representation
        notes_data = []
        tempo = 120  # Default tempo
        
        # Try to get tempo from the score
        try:
            # Look for tempo markings in the score
            tempo_markings = simplified_score.flat.getElementsByClass('MetronomeMark')
            if tempo_markings:
                tempo = tempo_markings[0].number
            else:
                # Alternative method to find tempo
                for element in simplified_score.recurse():
                    if hasattr(element, 'number') and hasattr(element, 'referent'):
                        tempo = element.number
                        break
        except:
            # If all else fails, use default tempo
            tempo = 120
        
        # Calculate time scaling based on tempo
        beat_duration = 60.0 / tempo  # Duration of one quarter note in seconds
        
        for element in simplified_score.recurse():
            try:
                if isinstance(element, note.Note):
                    # Get frequency from MIDI note
                    freq = element.pitch.frequency
                    start_time = float(element.offset) * beat_duration
                    duration = float(element.duration.quarterLength) * beat_duration
                    if start_time < duration_limit and freq > 0:  # Only include notes within time limit
                        notes_data.append((freq, start_time, min(duration, duration_limit - start_time)))
                elif isinstance(element, chord.Chord):
                    # For chords, take the root note
                    freq = element.root().frequency
                    start_time = float(element.offset) * beat_duration
                    duration = float(element.duration.quarterLength) * beat_duration
                    if start_time < duration_limit and freq > 0:  # Only include chords within time limit
                        notes_data.append((freq, start_time, min(duration, duration_limit - start_time)))
            except Exception as e:
                # Skip problematic elements
                continue
        
        if not notes_data:
            return None
        
        # Create audio from notes (simple synthesis)
        sample_rate = 22050
        audio_length = int(duration_limit * sample_rate)
        audio = np.zeros(audio_length)
        
        # Limit number of simultaneous notes for performance
        max_notes = min(100, len(notes_data))
        
        for freq, start_time, duration in notes_data[:max_notes]:
            if start_time >= duration_limit or duration <= 0:
                continue
                
            start_sample = int(start_time * sample_rate)
            duration_samples = int(duration * sample_rate)
            end_sample = min(start_sample + duration_samples, audio_length)
            
            if start_sample < end_sample and freq > 0 and freq < 8000:  # Reasonable frequency range
                # Generate sine wave
                t = np.linspace(0, (end_sample - start_sample) / sample_rate, end_sample - start_sample)
                # Apply envelope to reduce clicks
                envelope = np.exp(-t * 3) if len(t) > 0 else np.array([])
                wave = 0.1 * np.sin(2 * np.pi * freq * t) * envelope
                
                if start_sample < len(audio) and end_sample <= len(audio):
                    audio[start_sample:end_sample] += wave
        
        # Normalize audio and apply gentle limiting
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.7  # Leave some headroom
            # Apply soft clipping to prevent harsh distortion
            audio = np.tanh(audio * 2) * 0.8
        
        # Save as WAV file
        sf.write(str(audio_path), audio, sample_rate)
        
        return str(audio_path)
        
    except Exception as e:
        st.warning(f"Could not convert MIDI to audio: {str(e)}")
        return None

@st.cache_data
def load_exported_features():
    """Load the pre-exported features from data/features directory"""
    try:
        # Load pre-extracted features
        musical_df = pd.read_pickle(FEATURES_DIR / "musical_features_df.pkl")
        harmonic_df = pd.read_pickle(FEATURES_DIR / "harmonic_features_df.pkl")
        note_sequences = np.load(FEATURES_DIR / "note_sequences.npy")
        sequence_labels = np.load(FEATURES_DIR / "sequence_labels.npy")
        
        # Load note mapping if available
        try:
            with open(FEATURES_DIR / "note_mapping.pkl", 'rb') as f:
                note_mapping = pickle.load(f)
        except:
            note_mapping = None
        
        return {
            'musical_df': musical_df,
            'harmonic_df': harmonic_df,
            'note_sequences': note_sequences,
            'sequence_labels': sequence_labels,
            'note_mapping': note_mapping
        }
        
    except Exception as e:
        st.error(f"Error loading exported features: {str(e)}")
        return None

def predict_composer_with_progress(selected_file, composer_name, progress_bar, status_text):
    """
    Prediction function with progress updates using pre-loaded data.
    """
    try:
        # Step 1: Get exported features from session state (20%)
        progress_bar.progress(20)
        status_text.text("🔄 Preparing features...")
        time.sleep(0.5)  # Brief pause for UX
        
        exported_features = st.session_state.exported_features
        musical_df = exported_features['musical_df']
        harmonic_df = exported_features['harmonic_df']
        note_sequences = exported_features['note_sequences']
        sequence_labels = exported_features['sequence_labels']
        note_mapping = exported_features.get('note_mapping', {})
        
        # Step 2: Find the specific file (40%)
        progress_bar.progress(40)
        status_text.text("🔍 Finding file in feature database...")
        time.sleep(0.5)
        
        # Try to find the file in the musical features dataframe
        file_matches = []
        
        # Strategy 1: Exact filename match
        file_matches.extend(musical_df[musical_df['filename'] == selected_file].index.tolist())
        
        # Strategy 2: Check if the file path contains our file
        if not file_matches:
            file_matches.extend(musical_df[musical_df['file_path'].str.contains(selected_file, na=False)].index.tolist())
        
        # Strategy 3: Check for partial filename match (without extension)
        if not file_matches:
            base_name = selected_file.replace('.mid', '').replace('.midi', '')
            file_matches.extend(musical_df[musical_df['filename'].str.contains(base_name, na=False)].index.tolist())
        
        if not file_matches:
            return None, None, None, f"File '{selected_file}' not found in exported features"
        
        # Use the first match
        file_idx = file_matches[0]
        
        # Extract features for this specific file
        musical_row = musical_df.iloc[file_idx]
        
        # Find corresponding harmonic features (match by file index)
        if file_idx < len(harmonic_df):
            harmonic_row = harmonic_df.iloc[file_idx]
        else:
            # Find by filename if direct index doesn't work
            harmonic_matches = harmonic_df[harmonic_df['filename'] == selected_file]
            if len(harmonic_matches) > 0:
                harmonic_row = harmonic_matches.iloc[0]
            else:
                harmonic_row = None
        
        # Get musical features (exclude non-numeric columns)
        musical_numeric_cols = musical_df.select_dtypes(include=[np.number]).columns
        musical_numeric_cols = musical_numeric_cols.drop(['composer'], errors='ignore')
        musical_features = musical_row[musical_numeric_cols].values.reshape(1, -1)
        
        # Validate feature count - ensure we have 17 features as expected
        if musical_features.shape[1] != 17:
            # Pad with zeros if we have fewer features, truncate if we have more
            if musical_features.shape[1] < 17:
                padding = np.zeros((1, 17 - musical_features.shape[1]))
                musical_features = np.concatenate([musical_features, padding], axis=1)
            else:
                musical_features = musical_features[:, :17]
        
        # Get harmonic features (exclude non-numeric columns)
        if harmonic_row is not None:
            harmonic_numeric_cols = harmonic_df.select_dtypes(include=[np.number]).columns
            harmonic_numeric_cols = harmonic_numeric_cols.drop(['composer'], errors='ignore')
            harmonic_features = harmonic_row[harmonic_numeric_cols].values.reshape(1, -1)
        else:
            # If no harmonic features for this file, create zeros
            harmonic_numeric_cols = harmonic_df.select_dtypes(include=[np.number]).columns
            harmonic_numeric_cols = harmonic_numeric_cols.drop(['composer'], errors='ignore')
            harmonic_features = np.zeros((1, len(harmonic_numeric_cols)))
        
        # Get sequence features - find sequences from the same composer
        # Map composer name to integer using the mapping
        composer_to_int = note_mapping.get('composer_to_int', {'Bach': 0, 'Beethoven': 1, 'Chopin': 2, 'Mozart': 3})
        composer_int = composer_to_int.get(composer_name, 0)
        
        # Find sequences with matching composer
        composer_sequence_mask = sequence_labels == composer_int
        composer_sequences = note_sequences[composer_sequence_mask]
        
        if len(composer_sequences) > 0:
            # Use a random sequence from this composer for better diversity
            random_idx = np.random.randint(0, len(composer_sequences))
            sequence_features = composer_sequences[random_idx].reshape(1, -1)
        else:
            # Fallback: use first available sequence
            sequence_features = note_sequences[0].reshape(1, -1) if len(note_sequences) > 0 else np.zeros((1, 100))
        
        # Step 3: Get model from session state (60%)
        progress_bar.progress(60)
        status_text.text("🧠 Preparing AI model...")
        time.sleep(0.5)
        
        model = st.session_state.model
        
        # Check model input shapes and prepare features
        try:
            input_shapes = [input_layer.shape for input_layer in model.inputs]
            musical_dim = input_shapes[0][1] if len(input_shapes) > 0 else musical_features.shape[1]
            harmonic_dim = input_shapes[1][1] if len(input_shapes) > 1 else harmonic_features.shape[1]
            sequence_dim = input_shapes[2][1] if len(input_shapes) > 2 else sequence_features.shape[1]
        except Exception as e:
            # Use current dimensions
            musical_dim, harmonic_dim, sequence_dim = musical_features.shape[1], harmonic_features.shape[1], sequence_features.shape[1]
        
        # Prepare features for model input - pad or truncate to expected dimensions
        musical_final = np.zeros((1, musical_dim))
        harmonic_final = np.zeros((1, harmonic_dim))
        sequence_final = np.zeros((1, sequence_dim))
        
        # Copy available features
        musical_final[0, :min(musical_dim, musical_features.shape[1])] = musical_features[0, :min(musical_dim, musical_features.shape[1])]
        harmonic_final[0, :min(harmonic_dim, harmonic_features.shape[1])] = harmonic_features[0, :min(harmonic_dim, harmonic_features.shape[1])]
        sequence_final[0, :min(sequence_dim, sequence_features.shape[1])] = sequence_features[0, :min(sequence_dim, sequence_features.shape[1])]
        
        # Apply preprocessing to features (same as training)
        try:
            from sklearn.preprocessing import StandardScaler, RobustScaler
            
            # Use the training data to fit new scalers (same approach as notebook)
            # Get all training features to fit scalers properly
            all_musical = musical_df.select_dtypes(include=[np.number]).drop(['composer'], errors='ignore').values
            all_harmonic = harmonic_df.select_dtypes(include=[np.number]).drop(['composer'], errors='ignore').values
            
            # Musical features: RobustScaler (as used in notebook)
            musical_scaler = RobustScaler(quantile_range=(5.0, 95.0))
            musical_scaler.fit(all_musical)
            musical_final = musical_scaler.transform(musical_final)
            
            # Harmonic features: StandardScaler  
            harmonic_scaler = StandardScaler()
            harmonic_scaler.fit(all_harmonic)
            harmonic_final = harmonic_scaler.transform(harmonic_final)
            
            # Sequence features: StandardScaler
            sequence_scaler = StandardScaler()
            sequence_scaler.fit(note_sequences)
            sequence_final = sequence_scaler.transform(sequence_final)
            
        except Exception as e:
            # Continue without preprocessing if it fails
            pass
        
        # Step 4: Analyze patterns (80%)
        progress_bar.progress(80)
        status_text.text("🎵 Analyzing musical patterns...")
        time.sleep(0.5)
        
        # Make prediction
        prediction_probs = model.predict([musical_final, harmonic_final, sequence_final], verbose=0)[0]
        
        predicted_idx = np.argmax(prediction_probs)
        predicted_composer = TARGET_COMPOSERS[predicted_idx]
        confidence = prediction_probs[predicted_idx]
        
        # Get all probabilities
        all_probs = {TARGET_COMPOSERS[i]: prediction_probs[i] for i in range(len(TARGET_COMPOSERS))}
        
        # Step 5: Complete (100%)
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)
        
        return predicted_composer, confidence, all_probs, "Real Model (using exported features)"
        
    except Exception as e:
        return None, None, None, f"Error: {str(e)}"

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎼 Classical Composer Identifier</h1>
        <p>Discover the musical DNA of classical composers through AI analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load all application data at startup
    if 'app_data_loaded' not in st.session_state:
        model, exported_features, midi_files = load_all_app_data()
        
        if model is None or exported_features is None:
            st.error("Failed to load required application data. Please check the model and feature files.")
            return
        
        # Store in session state
        st.session_state.model = model
        st.session_state.exported_features = exported_features
        st.session_state.midi_files = midi_files
        st.session_state.app_data_loaded = True
    
    # Get data from session state
    model = st.session_state.model
    exported_features = st.session_state.exported_features
    midi_files = st.session_state.midi_files
    
    # Sidebar for composer selection
    st.sidebar.markdown("## 🎭 Select a Composer")
    selected_composer = st.sidebar.selectbox(
        "Choose a composer:",
        TARGET_COMPOSERS,
        help="Select a composer to view their works"
    )
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"## 🎼 {selected_composer}'s Musical Works")
        
        # Display composer info
        with st.expander(f"📖 About {selected_composer}", expanded=True):
            st.markdown(COMPOSER_INFO[selected_composer])
        
        # MIDI file selection
        if midi_files[selected_composer]:
            st.markdown("### 🎵 Available Compositions")
            selected_file = st.selectbox(
                f"Choose a {selected_composer} composition:",
                midi_files[selected_composer],
                help="Select a MIDI file to analyze"
            )
            
            if selected_file:
                # File info
                midi_file_path = MIDI_DIR / selected_composer / selected_file
                
                st.markdown(f"**Selected:** `{selected_file}`")
                
                # File details
                try:
                    file_size = midi_file_path.stat().st_size
                    st.markdown(f"**File size:** {file_size:,} bytes")
                except:
                    pass
                
                # Audio Preview Section
                st.markdown("### 🎵 Audio Preview")
                
                # Audio settings
                with st.expander("🎛️ Audio Settings", expanded=False):
                    preview_duration = st.slider(
                        "Preview Duration (seconds)", 
                        min_value=10, 
                        max_value=60, 
                        value=30, 
                        step=5,
                        help="Choose how long the audio preview should be"
                    )
                    
                    audio_quality = st.selectbox(
                        "Audio Quality",
                        options=["Standard (22kHz)", "High (44kHz)"],
                        index=0,
                        help="Higher quality uses more processing time"
                    )
                
                col_audio1, col_audio2 = st.columns([2, 1])
                
                with col_audio1:
                    # Generate audio preview with user settings
                    sample_rate = 44100 if "High" in audio_quality else 22050
                    
                    with st.spinner(f"🎼 Converting MIDI to audio ({preview_duration}s preview)..."):
                        audio_path = convert_midi_to_audio(midi_file_path, duration_limit=preview_duration)
                    
                    if audio_path and Path(audio_path).exists():
                        st.markdown(f"**🎧 Listen to a preview (first {preview_duration} seconds):**")
                        
                        # Display audio player
                        try:
                            audio_file = open(audio_path, 'rb')
                            audio_bytes = audio_file.read()
                            audio_file.close()
                            
                            st.audio(audio_bytes, format='audio/wav', start_time=0)
                            
                            # Show audio info
                            audio_size = len(audio_bytes)
                            st.caption(f"Preview audio: {audio_size:,} bytes • {preview_duration}s duration • Generated from MIDI")
                            
                            # Audio controls info
                            st.info("💡 **Tip:** Use the audio player controls to play, pause, and adjust volume. This is a simplified audio preview generated from MIDI data.")
                            
                        except Exception as e:
                            st.error(f"Error playing audio: {e}")
                            
                            # Offer alternative
                            st.markdown("**Alternative:** You can download the MIDI file and play it with your preferred MIDI player.")
                            
                    else:
                        st.warning("⚠️ Could not generate audio preview. This might be due to:")
                        st.markdown("""
                        - Complex MIDI structure that couldn't be simplified
                        - Missing or corrupted MIDI data
                        - Very short MIDI file with no playable notes
                        - Audio conversion limitations
                        
                        **Don't worry!** The MIDI file can still be analyzed for composer identification.
                        """)
                        
                        # Offer to try with a different file
                        st.info("💡 **Suggestion:** Try selecting a different composition - some MIDI files work better than others for audio preview.")
                
                with col_audio2:
                    st.markdown("**🎼 MIDI Info**")
                    
                    try:
                        # Try to get basic MIDI info using music21
                        score = converter.parse(str(midi_file_path))
                        
                        # Count different types of elements
                        try:
                            notes_count = len([n for n in score.flatten().notes if isinstance(n, note.Note)])
                            chords_count = len([c for c in score.flatten().notes if isinstance(c, chord.Chord)])
                            
                            st.metric("🎵 Notes", notes_count)
                            st.metric("🎼 Chords", chords_count)
                        except:
                            st.info("Note/chord count unavailable")
                        
                        # Try to get duration
                        try:
                            duration = score.duration.quarterLength
                            st.metric("⏱️ Duration (beats)", f"{duration:.1f}")
                        except:
                            st.info("Duration information unavailable")
                        
                        # Try to get key signature
                        try:
                            key = score.analyze('key')
                            if key:
                                st.markdown(f"**🗝️ Key:** {key}")
                            else:
                                st.info("Key signature not detected")
                        except:
                            st.info("Key analysis unavailable")
                        
                        # Try to get tempo information
                        try:
                            tempo_markings = score.flat.getElementsByClass('MetronomeMark')
                            if tempo_markings:
                                tempo = tempo_markings[0].number
                                st.markdown(f"**🎵 Tempo:** {tempo} BPM")
                            else:
                                st.markdown(f"**🎵 Tempo:** Default (120 BPM)")
                        except:
                            st.info("Tempo information unavailable")
                        
                    except Exception as e:
                        st.info("MIDI analysis not available")
                        st.caption(f"Technical details: {str(e)[:100]}...")
                
                # Prediction section
                st.markdown("### 🔍 AI Composer Analysis")
                
                if st.button("🎯 Identify Composer", type="primary", use_container_width=True):
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Run prediction with progress updates
                        result = predict_composer_with_progress(selected_file, selected_composer, progress_bar, status_text)
                        
                        if result[0] is not None:
                            predicted_composer, confidence, all_probs, method = result
                            # Store results in session state
                            st.session_state.prediction_results = {
                                'predicted_composer': predicted_composer,
                                'confidence': confidence,
                                'all_probs': all_probs,
                                'true_composer': selected_composer,
                                'file_name': selected_file,
                                'method': method
                            }
                            
                            # Clear progress indicators after success
                            time.sleep(0.5)  # Brief pause to show completion
                            progress_bar.empty()
                            status_text.empty()
                        else:
                            progress_bar.empty()
                            status_text.empty()
                            st.error("Prediction failed. Please try again.")
                            
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        progress_bar.empty()
                        status_text.empty()
                
                # Display prediction results
                if hasattr(st.session_state, 'prediction_results'):
                    results = st.session_state.prediction_results
                    
                    st.markdown("### 🎯 Prediction Results")
                    
                    # Check if prediction is correct
                    is_correct = results['predicted_composer'] == results['true_composer']
                    
                    # Main prediction result
                    if is_correct:
                        st.markdown(f"""
                        <div class="prediction-result">
                            <h2>✅ Correct Prediction!</h2>
                            <h3>🎼 Predicted: {results['predicted_composer']}</h3>
                            <p>Confidence: {results['confidence']:.2%}</p>
                            <p><small>Method: {results.get('method', 'Unknown')}</small></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-result">
                            <h2>❌ Incorrect Prediction</h2>
                            <h3>🎼 Predicted: {results['predicted_composer']}</h3>
                            <h3>🎯 Actual: {results['true_composer']}</h3>
                            <p>Confidence: {results['confidence']:.2%}</p>
                            <p><small>Method: {results.get('method', 'Unknown')}</small></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Probability breakdown
                    st.markdown("#### 📊 Confidence Scores")
                    
                    # Create columns for each composer
                    prob_cols = st.columns(4)
                    
                    for i, (composer, prob) in enumerate(results['all_probs'].items()):
                        with prob_cols[i]:
                            is_predicted = composer == results['predicted_composer']
                            st.metric(
                                label=f"🎭 {composer}",
                                value=f"{prob:.2%}",
                                delta=f"{'✅ Predicted' if is_predicted else ''}"
                            )
                    
                    # Visualization
                    st.markdown("#### 📈 Prediction Visualization")
                    
                    # Create bar chart
                    prob_df = pd.DataFrame({
                        'Composer': list(results['all_probs'].keys()),
                        'Probability': list(results['all_probs'].values())
                    })
                    
                    st.bar_chart(prob_df.set_index('Composer'))
        
        else:
            st.warning(f"No MIDI files found for {selected_composer}")
    
    with col2:
        st.markdown(f"## 🖼️ {selected_composer}")
        
        # Display composer image
        composer_image = load_composer_image(selected_composer)
        if composer_image:
            st.image(composer_image, caption=f"{selected_composer}", use_container_width=True)
        else:
            st.info("Image not available")
        
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        
        if midi_files[selected_composer]:
            st.metric("🎵 Available Works", len(midi_files[selected_composer]))
        
        # Model info
        st.markdown("### 🤖 AI Model Info")
        
        if model:
            try:
                total_params = model.count_params()
                st.metric("🧠 Model Parameters", f"{total_params:,}")
                
                # Show model input shapes
                input_shapes = [input_layer.shape for input_layer in model.inputs]
                st.markdown(f"**Input Shapes:** {len(input_shapes)} inputs")
                    
            except Exception as e:
                st.warning(f"Could not get model info: {e}")
        
        st.metric("🎯 Target Composers", len(TARGET_COMPOSERS))
        
        # Legend
        st.markdown("### 🎭 Composers")
        for composer in TARGET_COMPOSERS:
            st.markdown(f"• **{composer}** 🎼")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🎼 Built with Streamlit • Powered by Deep Learning • 🎵</p>
        <p>Neural Networks and Deep Learning Project - AAI-511</p>
        <p><small>University of San Diego - Section 5, Group 2</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
