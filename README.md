# University of San Diego  
## AAI-511: Neural Networks and Deep Learning

**Professor**: Rod Albuyeh  
**Section**: 5  
**Group**: 2  
**Contributors**:
- Ashley Moore
- Kevin Pooler
- Swapnil Patil

# From Notes to Networks: Decoding the Musical DNA of Classical Composers through Symbolic Sequence Learning
### Composer Classification Using Deep Learning

## Overview
This project develops a deep learning model to predict the composer of classical music pieces using neural network techniques. The system analyzes MIDI files from renowned composers and uses both LSTM and CNN architectures to identify unique compositional patterns and styles.

## Problem Statement / Objective / Use Cases

### Problem Statement
Identifying the composer of classical music pieces can be challenging, especially for novice musicians or listeners. While expert musicologists can recognize compositional styles, this requires extensive training and experience. An automated system could democratize this knowledge and assist in music education and analysis.

### Objective
- Develop a deep learning model that can accurately predict the composer of a given musical score
- Utilize Long Short-Term Memory (LSTM) and Convolutional Neural Network (CNN) techniques for music analysis
- Create an interactive web application for real-time composer prediction
- Analyze musical features and patterns that distinguish different composers

### Use Cases
- **Music Education:** Help students learn to identify different compositional styles and periods
- **Music Analysis:** Assist musicologists in analyzing and categorizing musical works
- **Digital Music Libraries:** Automatically tag and organize classical music collections
- **Music Discovery:** Help listeners explore works by specific composers based on style preferences
- **Academic Research:** Support research in computational musicology and pattern recognition

## Demo

[🎼 Interactive Composer Prediction App](https://your-app-url.streamlit.app/)

## Detailed Document

https://swapnilprakashpatil.github.io/aai511_2proj/FinalProject-7.2-Section%205-Team%202.html

[Complete Project Notebook](./FinalProject-7.2-Section%205-Team%202.ipynb)

## Project Architecture

```
aai511_2proj/
│
├── data/
│   ├── header.png                      # Project header image
│   ├── features/                       # Processed musical features
│   │   ├── harmonic_features_df.pkl
│   │   ├── musical_features_df.pkl
│   │   ├── note_mapping.pkl
│   │   ├── note_sequences.npy
│   │   └── sequence_labels.npy
│   ├── midiclassics/                   # Raw MIDI datasets
│   │   ├── Bach/                       # J.S. Bach compositions (50+ pieces)
│   │   ├── Beethoven/                  # Ludwig van Beethoven works
│   │   ├── Chopin/                     # Frédéric Chopin pieces
│   │   └── Mozart/                     # Wolfgang Amadeus Mozart compositions
│   └── model/                          # Trained model artifacts
│       ├── composer_classification_model_artifacts.pkl
│       ├── composer_classification_model_best.keras
│       ├── composer_classification_model_history.pkl
│       └── composer_classification_model.keras.keras
│
├── notebooks/
│   ├── FinalProject-7.2-Section 5-Team 2.ipynb  # Main project notebook
│   └── Hybrid.ipynb                             # Additional experiments
│
├── app.py                              # Streamlit web application
├── requirements.txt                    # Project dependencies
├── streamlit_requirements.txt          # Streamlit-specific dependencies
└── README.md
```

## Key Features

### Deep Learning Architecture
- **LSTM Networks:** Captures sequential patterns and temporal dependencies in musical compositions
- **CNN Models:** Analyzes local musical patterns and harmonic structures
- **Hybrid Approach:** Combines both architectures for enhanced prediction accuracy
- **Feature Engineering:** Extracts harmonic features, note sequences, and musical characteristics

### Composer Coverage
- **Johann Sebastian Bach** (1685-1750): Baroque period, known for complex fugues
- **Ludwig van Beethoven** (1770-1827): Classical to Romantic transition, symphonic works
- **Wolfgang Amadeus Mozart** (1756-1791): Classical period, elegant compositions
- **Frédéric Chopin** (1810-1849): Romantic period, virtuosic piano works

### Interactive Web Application
- Real-time MIDI file upload and analysis
- Composer prediction with confidence scores
- Visual composer information with historical context
- Sample audio playback capabilities
- Responsive design with intuitive interface

### Musical Feature Analysis
- Note sequence extraction from MIDI files
- Harmonic progression analysis
- Temporal pattern recognition
- Rhythmic structure identification
- Melodic contour analysis

## Technical Implementation

### Data Processing Pipeline
1. **MIDI File Parsing:** Extract musical events using music21 and pretty_midi libraries
2. **Feature Extraction:** Convert musical data into numerical sequences
3. **Sequence Preparation:** Create fixed-length input sequences for neural networks
4. **Data Augmentation:** Apply musical transformations to increase dataset diversity

### Model Architecture
- **Input Layer:** Processes note sequences and musical features
- **LSTM Layers:** Captures temporal dependencies in musical phrases
- **CNN Layers:** Identifies local patterns and harmonic structures
- **Dense Layers:** Combines features for final classification
- **Output Layer:** 4-class softmax for composer prediction

### Performance Metrics
- Classification accuracy across all composers
- Confusion matrix analysis for error patterns
- Precision and recall per composer
- Cross-validation for model robustness

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/swapnilprakashpatil/aai511_2proj.git
   cd aai511_2proj
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter notebook:**
   ```bash
   jupyter notebook "FinalProject-7.2-Section 5-Team 2.ipynb"
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Dataset Information

### Source
- Classical MIDI dataset containing works from four major composers
- Over 200 MIDI files representing different musical periods and styles
- Files sourced from public domain classical music repositories

### Preprocessing
- MIDI files converted to note sequences and harmonic features
- Sequences normalized and padded for consistent input dimensions
- Labels encoded for multi-class classification
- Train/validation/test splits for proper model evaluation

## Model Performance

The trained model achieves:
- **Overall Accuracy:** 85%+ on test dataset
- **Bach:** High precision due to distinctive fugal patterns
- **Beethoven:** Good recognition of dynamic contrasts and developmental techniques
- **Mozart:** Strong identification of classical form structures
- **Chopin:** Excellent recognition of romantic harmonic progressions

## Future Enhancements

- Expand to additional composers and musical periods
- Include audio feature extraction alongside MIDI analysis
- Implement real-time audio recording and analysis
- Add detailed musical analysis explanations
- Support for ensemble and orchestral works
- Integration with digital music streaming platforms

## Technologies Used

- **Deep Learning:** TensorFlow/Keras for neural network implementation
- **Music Processing:** music21, pretty_midi for MIDI analysis
- **Web Framework:** Streamlit for interactive application
- **Data Science:** NumPy, Pandas for data manipulation
- **Visualization:** Matplotlib, Plotly for data visualization
- **Audio Processing:** librosa for advanced audio features

## License

This project is developed for educational purposes as part of the AAI-511 course at the University of San Diego.