# Design Document: FractalLens

## Overview

FractalLens is a diagnostic AI system that leverages physics-based feature extraction and Physics-Informed Neural Networks (PINNs) to detect pathologies in low-resolution medical imaging. The system is architected for edge deployment in rural healthcare settings with intermittent connectivity.

The core innovation lies in combining Fractal Dimension Analysis with PINNs to achieve high diagnostic accuracy on noisy, low-resolution images without requiring cloud GPU infrastructure. The system operates offline-first with opportunistic synchronization, ensuring continuous clinical operations regardless of network availability.

### Key Design Principles

1. **Physics-Informed Processing**: Incorporate domain knowledge from medical imaging physics into both feature extraction and neural network architecture
2. **Edge-First Architecture**: Optimize for CPU-only inference with minimal memory footprint
3. **Offline Resilience**: Design all core functionality to work without internet connectivity
4. **Explainability by Design**: Provide visual and quantitative explanations for all diagnostic predictions
5. **Simplicity**: Create an intuitive interface suitable for users with minimal technical training

## Architecture

The system follows a modular architecture with clear separation between image processing, inference, data management, and presentation layers.

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Streamlit UI (Explainability Dashboard)       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   Image      │  │  Inference   │  │  Explainability │  │
│  │ Preprocessor │  │   Engine     │  │    Generator    │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                      Core ML Layer                           │
│  ┌──────────────────────┐  ┌──────────────────────────┐    │
│  │  Physics Feature     │  │     PINN Model           │    │
│  │    Extractor         │  │  (PyTorch)               │    │
│  │  - Fractal Dimension │  │  - Pathology Detection   │    │
│  │  - Lacunarity        │  │  - Physics Constraints   │    │
│  └──────────────────────┘  └──────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   SQLite     │  │    Sync      │  │   File Storage  │  │
│  │   Database   │  │   Manager    │  │   (Images)      │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

1. **Image Loading**: User uploads medical image through Streamlit UI
2. **Preprocessing**: Image Preprocessor validates, normalizes, and prepares image
3. **Feature Extraction**: Physics Feature Extractor calculates FD and Lacunarity
4. **Inference**: PINN Model processes features and generates predictions
5. **Explainability**: Explainability Generator creates heatmaps and identifies high-entropy regions
6. **Presentation**: Dashboard displays results with visual explanations
7. **Persistence**: Local Database stores all data; Sync Manager handles opportunistic uploads

## Components and Interfaces

### 1. Image Preprocessor

**Responsibility**: Load, validate, and prepare medical images for analysis.

**Interface**:
```python
class ImagePreprocessor:
    def load_image(file_path: str) -> ImageData:
        """
        Load medical image from file.
        
        Args:
            file_path: Path to image file (DICOM, PNG, JPEG)
            
        Returns:
            ImageData object containing pixel array and metadata
            
        Raises:
            InvalidImageError: If image format is unsupported or corrupted
            ResolutionError: If image resolution is below 256x256
        """
        
    def preprocess(image: ImageData) -> ProcessedImage:
        """
        Preprocess image for feature extraction.
        
        Steps:
        1. Convert to grayscale if needed
        2. Normalize pixel intensities to [0, 1]
        3. Apply Gaussian noise reduction (sigma=1.0)
        4. Resize to standard dimensions if needed
        
        Args:
            image: Raw image data
            
        Returns:
            ProcessedImage ready for feature extraction
        """
        
    def extract_dicom_metadata(image: ImageData) -> DicomMetadata:
        """Extract patient ID, scan date, modality from DICOM files."""
```

**Implementation Notes**:
- Use OpenCV for image loading and preprocessing
- Support DICOM via pydicom library
- Implement validation checks for resolution and format
- Cache preprocessed images to avoid redundant computation

### 2. Physics Feature Extractor

**Responsibility**: Calculate Fractal Dimension and Lacunarity metrics from medical images.

**Interface**:
```python
class PhysicsFeatureExtractor:
    def calculate_fractal_dimension(image: ProcessedImage, 
                                   method: str = "box_counting") -> float:
        """
        Calculate Fractal Dimension using box-counting method.
        
        The box-counting method:
        1. Overlay grids of decreasing box sizes on the image
        2. Count boxes containing image features at each scale
        3. Calculate FD from log-log slope of count vs box size
        
        Args:
            image: Preprocessed medical image
            method: Algorithm to use (box_counting, differential_box_counting)
            
        Returns:
            Fractal Dimension value (typically 1.0 to 3.0)
        """
        
    def calculate_lacunarity(image: ProcessedImage, 
                            box_sizes: List[int] = [2, 4, 8, 16]) -> float:
        """
        Calculate Lacunarity metric measuring texture heterogeneity.
        
        Lacunarity quantifies gap distribution:
        1. Apply gliding box algorithm at multiple scales
        2. Calculate mass distribution variance at each scale
        3. Compute lacunarity from variance/mean ratio
        
        Args:
            image: Preprocessed medical image
            box_sizes: List of box sizes for multi-scale analysis
            
        Returns:
            Lacunarity value (higher = more heterogeneous texture)
        """
        
    def extract_regional_features(image: ProcessedImage, 
                                  grid_size: int = 8) -> RegionalFeatures:
        """
        Calculate FD and Lacunarity for image regions.
        
        Divides image into grid_size x grid_size regions and calculates
        features for each region to enable spatial anomaly detection.
        
        Args:
            image: Preprocessed medical image
            grid_size: Number of regions per dimension
            
        Returns:
            RegionalFeatures containing FD and Lacunarity per region
        """
```

**Implementation Notes**:
- Implement box-counting algorithm with logarithmic scale progression
- Use sliding window approach for lacunarity calculation
- Optimize for CPU execution using NumPy vectorization
- Ensure numerical stability for noisy images
- Target computation time: < 2 seconds per image

### 3. PINN Model

**Responsibility**: Perform pathology detection using physics-informed neural network.

**Architecture**:
```
Input Layer (Feature Vector)
    ↓
Dense Layer (256 units, ReLU)
    ↓
Physics Constraint Layer (custom)
    ↓
Dense Layer (128 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense Layer (64 units, ReLU)
    ↓
Output Layer (3 units, Softmax)
    ↓
[Tumor, Retinal Issue, Normal]
```

**Interface**:
```python
class PINNModel:
    def __init__(model_path: str):
        """Load pretrained PINN model from disk."""
        
    def predict(features: FeatureVector) -> Prediction:
        """
        Generate pathology prediction from extracted features.
        
        Args:
            features: Combined global and regional physics features
            
        Returns:
            Prediction containing:
            - class_probabilities: Dict[str, float] for each pathology class
            - confidence: Overall prediction confidence (0-1)
            - high_entropy_regions: List of (x, y, entropy_score) tuples
            - requires_review: Boolean flag if confidence < 0.7
        """
        
    def get_attention_weights(features: FeatureVector) -> AttentionMap:
        """
        Extract attention weights showing which features influenced prediction.
        
        Returns:
            AttentionMap indicating feature importance for explainability
        """
```

**Physics Constraints**:
The PINN incorporates physics constraints through a custom layer that enforces:
1. **Monotonicity**: Higher FD values in a region should correlate with higher anomaly probability
2. **Spatial Coherence**: Adjacent regions with similar features should have similar predictions
3. **Scale Invariance**: Predictions should be robust to small variations in image scale

**Training Considerations**:
- Loss function combines cross-entropy with physics constraint penalties
- Training data augmentation includes noise injection and resolution variation
- Model quantization for reduced size and faster CPU inference
- Target model size: < 500MB, inference time: < 10 seconds on CPU

### 4. Inference Engine

**Responsibility**: Orchestrate the complete diagnostic pipeline from image to prediction.

**Interface**:
```python
class InferenceEngine:
    def __init__(model: PINNModel, 
                 preprocessor: ImagePreprocessor,
                 feature_extractor: PhysicsFeatureExtractor):
        """Initialize inference engine with required components."""
        
    def diagnose(image_path: str) -> DiagnosticResult:
        """
        Execute complete diagnostic pipeline.
        
        Pipeline steps:
        1. Load and preprocess image
        2. Extract global physics features
        3. Extract regional physics features
        4. Run PINN inference
        5. Generate explainability data
        
        Args:
            image_path: Path to medical image file
            
        Returns:
            DiagnosticResult containing prediction, confidence, 
            heatmap data, and feature values
            
        Raises:
            ProcessingError: If any pipeline step fails
        """
        
    def batch_diagnose(image_paths: List[str]) -> List[DiagnosticResult]:
        """Process multiple images sequentially."""
```

**Performance Optimization**:
- Implement result caching to avoid reprocessing identical images
- Use memory-mapped arrays for large image batches
- Monitor memory usage and implement garbage collection between images

### 5. Explainability Generator

**Responsibility**: Create visual and quantitative explanations for diagnostic predictions.

**Interface**:
```python
class ExplainabilityGenerator:
    def generate_heatmap(image: ProcessedImage, 
                        regional_features: RegionalFeatures,
                        prediction: Prediction) -> Heatmap:
        """
        Generate heatmap overlay showing anomaly probability by region.
        
        Algorithm:
        1. Calculate entropy score for each region based on FD/Lacunarity
        2. Normalize scores to [0, 1] range
        3. Apply Gaussian smoothing for visual continuity
        4. Map scores to color gradient (blue=low, red=high)
        
        Args:
            image: Original processed image
            regional_features: Physics features per region
            prediction: Model prediction with attention weights
            
        Returns:
            Heatmap as RGBA array for overlay on original image
        """
        
    def identify_high_entropy_regions(regional_features: RegionalFeatures,
                                     threshold: float = 0.7) -> List[Region]:
        """
        Identify regions with high entropy scores.
        
        Args:
            regional_features: Physics features per region
            threshold: Minimum entropy score to flag region
            
        Returns:
            List of Region objects with coordinates and scores
        """
        
    def generate_feature_summary(features: FeatureVector) -> FeatureSummary:
        """Create human-readable summary of extracted features."""
```

**Visualization Design**:
- Use perceptually uniform color maps (viridis or plasma)
- Ensure heatmaps are interpretable in grayscale for printing
- Provide adjustable opacity for overlay blending
- Include scale bar and legend in all visualizations

### 6. Local Database

**Responsibility**: Persist patient data, images, and diagnostic results locally.

**Schema**:
```sql
-- Patients table
CREATE TABLE patients (
    patient_id TEXT PRIMARY KEY,
    name_encrypted BLOB,
    age INTEGER,
    gender TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Images table
CREATE TABLE images (
    image_id TEXT PRIMARY KEY,
    patient_id TEXT,
    image_path TEXT,
    image_type TEXT,  -- CT, OCT
    modality TEXT,
    scan_date DATE,
    resolution TEXT,
    file_size_bytes INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- Diagnostic results table
CREATE TABLE diagnostic_results (
    result_id TEXT PRIMARY KEY,
    image_id TEXT,
    model_version TEXT,
    prediction_class TEXT,
    confidence_score REAL,
    tumor_probability REAL,
    retinal_probability REAL,
    normal_probability REAL,
    fractal_dimension REAL,
    lacunarity REAL,
    requires_review BOOLEAN,
    processing_time_seconds REAL,
    clinician_feedback TEXT,  -- correct, incorrect, uncertain
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    synced BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (image_id) REFERENCES images(image_id)
);

-- High entropy regions table
CREATE TABLE high_entropy_regions (
    region_id TEXT PRIMARY KEY,
    result_id TEXT,
    x_coordinate INTEGER,
    y_coordinate INTEGER,
    width INTEGER,
    height INTEGER,
    entropy_score REAL,
    FOREIGN KEY (result_id) REFERENCES diagnostic_results(result_id)
);

-- Audit log table
CREATE TABLE audit_log (
    log_id TEXT PRIMARY KEY,
    user_id TEXT,
    action TEXT,
    table_name TEXT,
    record_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Interface**:
```python
class LocalDatabase:
    def __init__(db_path: str = "fractallens.db"):
        """Initialize SQLite database connection."""
        
    def save_diagnostic_result(result: DiagnosticResult) -> str:
        """
        Persist diagnostic result to database.
        
        Returns:
            result_id for the saved record
        """
        
    def get_patient_history(patient_id: str) -> List[DiagnosticResult]:
        """Retrieve all diagnostic results for a patient."""
        
    def mark_synced(result_ids: List[str]):
        """Mark results as synchronized to remote server."""
        
    def get_unsynced_results() -> List[DiagnosticResult]:
        """Retrieve all results not yet synced."""
        
    def log_audit_event(user_id: str, action: str, record_id: str):
        """Record audit event for compliance."""
```

**Implementation Notes**:
- Use SQLite WAL mode for better concurrency
- Implement connection pooling for multi-threaded access
- Store images as file paths, not BLOBs, for better performance
- Encrypt sensitive fields using Fernet (symmetric encryption)
- Implement automatic backup every 24 hours

### 7. Sync Manager

**Responsibility**: Synchronize local data to remote server when connectivity is available.

**Interface**:
```python
class SyncManager:
    def __init__(remote_url: str, api_key: str):
        """Initialize sync manager with remote endpoint."""
        
    def check_connectivity() -> bool:
        """Test if remote server is reachable."""
        
    def sync_results(results: List[DiagnosticResult]) -> SyncStatus:
        """
        Upload diagnostic results to remote server.
        
        Process:
        1. Check connectivity
        2. Encrypt data for transmission
        3. Upload in batches of 10 results
        4. Handle conflicts (prefer most recent timestamp)
        5. Mark successfully synced results in local DB
        6. Implement exponential backoff on failures
        
        Args:
            results: List of diagnostic results to sync
            
        Returns:
            SyncStatus with success count, failure count, conflicts
        """
        
    def auto_sync():
        """
        Background task that periodically checks for connectivity
        and syncs unsynced results.
        
        Runs every 5 minutes when connectivity is available.
        """
```

**Sync Protocol**:
- Use HTTPS with TLS 1.3 for all communications
- Implement JWT-based authentication
- Use chunked transfer encoding for large image uploads
- Store sync checkpoints to enable resume after interruption
- Log all sync operations for troubleshooting

### 8. Explainability Dashboard (Streamlit UI)

**Responsibility**: Provide user interface for all system interactions.

**Pages**:

1. **Home Page**
   - Upload new image button
   - View patient history button
   - System status indicators (DB size, model version, sync status)

2. **Image Upload Page**
   - File upload widget (drag-and-drop)
   - Image preview
   - Patient information form
   - Process button

3. **Results Page**
   - Original image display
   - Heatmap overlay with opacity slider
   - Prediction probabilities (bar chart)
   - Confidence score (gauge)
   - Feature values (FD, Lacunarity) in table
   - High-entropy regions list with click-to-zoom
   - Export report button
   - Save to database button

4. **Patient History Page**
   - Patient search/filter
   - Results table with sorting
   - Click to view detailed results

5. **Settings Page**
   - Language selection (English/Hindi)
   - Model update interface
   - Database management (backup, cleanup)
   - Sync configuration

**Interface Design Principles**:
- Large, touch-friendly buttons for tablet use
- High contrast colors for visibility in bright clinic environments
- Minimal text, maximum visual communication
- Progress indicators for all long-running operations
- Offline mode indicator prominently displayed

## Data Models

### ImageData
```python
@dataclass
class ImageData:
    pixel_array: np.ndarray  # 2D array of pixel intensities
    width: int
    height: int
    format: str  # DICOM, PNG, JPEG
    metadata: Dict[str, Any]  # DICOM tags or EXIF data
```

### ProcessedImage
```python
@dataclass
class ProcessedImage:
    pixel_array: np.ndarray  # Normalized, grayscale, denoised
    original_dimensions: Tuple[int, int]
    preprocessing_steps: List[str]  # Audit trail
```

### FeatureVector
```python
@dataclass
class FeatureVector:
    global_fractal_dimension: float
    global_lacunarity: float
    regional_features: np.ndarray  # Shape: (grid_size, grid_size, 2)
    feature_extraction_time: float
```

### Prediction
```python
@dataclass
class Prediction:
    class_probabilities: Dict[str, float]  # {tumor, retinal, normal}
    predicted_class: str
    confidence: float
    requires_review: bool
    high_entropy_regions: List[Tuple[int, int, float]]  # (x, y, score)
    attention_weights: np.ndarray
```

### DiagnosticResult
```python
@dataclass
class DiagnosticResult:
    result_id: str
    patient_id: str
    image_id: str
    image_path: str
    features: FeatureVector
    prediction: Prediction
    heatmap: np.ndarray  # RGBA array
    model_version: str
    processing_time: float
    timestamp: datetime
    clinician_feedback: Optional[str] = None
```

### SyncStatus
```python
@dataclass
class SyncStatus:
    total_results: int
    synced_successfully: int
    failed: int
    conflicts: int
    last_sync_time: datetime
    error_messages: List[str]
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Physics Feature Extraction Properties

**Property 1: Complete Feature Extraction**
*For any* valid medical image, the Physics_Feature_Extractor should calculate and return both Fractal Dimension and Lacunarity values along with regional features for all image regions.
**Validates: Requirements 1.1, 1.2, 1.5**

**Property 2: Feature Calculation Stability**
*For any* medical image, calculating features multiple times should produce values with variance less than 5% across all repetitions.
**Validates: Requirements 1.4**

**Property 3: Pixel Intensity Normalization**
*For any* CT or OCT image, after preprocessing, all pixel intensities should be in the normalized range [0, 1].
**Validates: Requirements 1.6**

### PINN Model and Inference Properties

**Property 4: Prediction Completeness**
*For any* valid feature vector, the PINN_Model should generate a prediction containing probability scores for all three pathology classes (tumor, retinal issue, normal), a confidence score, and identified high-entropy regions.
**Validates: Requirements 2.1, 2.2, 2.4, 12.1**

**Property 5: Low Confidence Review Flagging**
*For any* prediction, if the confidence score is below 0.7, then the requires_review flag should be set to True.
**Validates: Requirements 2.5**

**Property 6: Batch Processing Robustness**
*For any* list of valid images, processing them sequentially should complete without crashes or memory overflow, producing a result for each image.
**Validates: Requirements 3.5**

### Explainability Properties

**Property 7: Results Display Completeness**
*For any* diagnostic result, the Explainability_Dashboard should display the original image, heatmap overlay, probability scores for all three classes, and the calculated Fractal Dimension and Lacunarity values.
**Validates: Requirements 4.1, 4.2, 4.4, 4.5**

**Property 8: Heatmap Color Mapping**
*For any* heatmap, regions with higher entropy scores should be mapped to warmer colors (higher values in the red channel) than regions with lower entropy scores.
**Validates: Requirements 4.3**

**Property 9: Regional Feature Display on Interaction**
*For any* high-entropy region, when clicked in the UI, detailed feature values for that specific region should be displayed.
**Validates: Requirements 4.6**

### Data Persistence Properties

**Property 10: Offline Data Persistence**
*For any* diagnostic result, saving to the Local_Database should succeed without requiring internet connectivity, and the saved data should be immediately retrievable.
**Validates: Requirements 5.2**

**Property 11: Image Compression**
*For any* medical image stored in the Local_Database, the stored file size should be less than the original file size while maintaining diagnostic quality.
**Validates: Requirements 5.4**

**Property 12: Referential Integrity**
*For any* database operation, foreign key constraints between patients, images, and diagnostic results should be enforced, preventing orphaned records.
**Validates: Requirements 5.5**

### Synchronization Properties

**Property 13: Automatic Sync Initiation**
*For any* system state with unsynced diagnostic records, when internet connectivity is detected, the Sync_Manager should automatically initiate synchronization.
**Validates: Requirements 6.1**

**Property 14: Complete Data Upload**
*For any* sync operation, all three data types (patient records, images, and diagnostic results) should be uploaded to the remote server.
**Validates: Requirements 6.2**

**Property 15: Sync Resumability**
*For any* sync operation interrupted by network loss, when connectivity returns, the sync should resume from the last successful checkpoint without re-uploading already synced data.
**Validates: Requirements 6.3**

**Property 16: Sync Deduplication**
*For any* diagnostic result successfully synced, the record should be marked as synchronized in the local database, and subsequent sync operations should not re-upload it.
**Validates: Requirements 6.4**

**Property 17: Conflict Resolution**
*For any* sync conflict between local and remote data, the record with the most recent timestamp should be retained, and the conflict should be logged for manual review.
**Validates: Requirements 6.5**

**Property 18: Encrypted Transmission**
*For any* data transmitted during synchronization, the connection should use TLS 1.3 or higher encryption.
**Validates: Requirements 6.6, 11.5**

### Image Processing Properties

**Property 19: Multi-Format Support**
*For any* valid image file in DICOM, PNG, or JPEG format, the FractalLens_System should successfully load and process it.
**Validates: Requirements 7.1**

**Property 20: Resolution Validation**
*For any* image with resolution below 256x256 pixels, the FractalLens_System should reject it with a clear error message explaining the minimum resolution requirement.
**Validates: Requirements 7.2, 7.3**

**Property 21: Noise Reduction Application**
*For any* medical image, the preprocessing pipeline should apply Gaussian noise reduction filters before feature extraction.
**Validates: Requirements 7.4**

**Property 22: Grayscale Conversion**
*For any* color image (3-channel RGB), the preprocessing should convert it to grayscale (single channel) for processing.
**Validates: Requirements 7.5**

**Property 23: DICOM Metadata Extraction**
*For any* DICOM file, the system should extract and make available patient ID, scan date, and modality metadata.
**Validates: Requirements 7.6**

### User Interface Properties

**Property 24: Image Preview Display**
*For any* image loaded by a user, the Explainability_Dashboard should display a preview before processing begins.
**Validates: Requirements 8.2**

**Property 25: Progress Indication**
*For any* processing operation, the Explainability_Dashboard should display a progress indicator showing the current step.
**Validates: Requirements 8.3**

**Property 26: Automatic Navigation**
*For any* completed processing operation, the Explainability_Dashboard should automatically navigate to the results view.
**Validates: Requirements 8.4**

**Property 27: Export Options Availability**
*For any* results view, the Explainability_Dashboard should provide options to save, print, and export the diagnostic report.
**Validates: Requirements 8.5**

### Model Management Properties

**Property 28: Model Update Mechanism**
*For any* new model file provided, the FractalLens_System should validate compatibility and either load the new model or retain the previous model with an error message.
**Validates: Requirements 9.1, 9.2, 9.3**

**Property 29: Model Version Logging**
*For any* successful model load, the system should create a log entry containing the model version and update timestamp.
**Validates: Requirements 9.4**

**Property 30: Model Version Audit Trail**
*For any* diagnostic result, the model version used should be recorded and retrievable for audit purposes.
**Validates: Requirements 9.5**

### Error Handling Properties

**Property 31: Error Logging and User Notification**
*For any* unexpected error during image processing, the system should both log the error details and display a user-friendly message.
**Validates: Requirements 10.1**

**Property 32: Corrupted File Handling**
*For any* batch of images containing corrupted files, the system should skip corrupted files and continue processing valid images.
**Validates: Requirements 10.2**

**Property 33: Database Operation Retry**
*For any* failed database operation, the system should retry up to 3 times before reporting failure to the user.
**Validates: Requirements 10.3**

**Property 34: Disk Space Protection**
*For any* disk full condition, the system should prevent new data writes and immediately alert the user.
**Validates: Requirements 10.4**

**Property 35: Crash Recovery**
*For any* application crash, unsaved diagnostic results should be preserved in a recovery file and be recoverable on restart.
**Validates: Requirements 10.5**

**Property 36: Operation Audit Logging**
*For any* system operation, a detailed log entry should be created for troubleshooting and audit purposes.
**Validates: Requirements 10.6**

### Security Properties

**Property 37: Sensitive Data Encryption**
*For any* patient data stored in the Local_Database, sensitive fields should be encrypted using AES-256 encryption.
**Validates: Requirements 11.1**

**Property 38: Session Timeout**
*For any* user session, after 15 minutes of inactivity, the system should automatically log out the user.
**Validates: Requirements 11.3**

**Property 39: Report Anonymization Option**
*For any* diagnostic report export operation, the system should provide an option to anonymize patient identifiers.
**Validates: Requirements 11.4**

**Property 40: Data Access Audit Trail**
*For any* data access or modification operation, an audit log entry should be created with user attribution.
**Validates: Requirements 11.6**

### Performance Monitoring Properties

**Property 41: Accuracy Metrics Calculation**
*For any* set of predictions where ground truth labels are available, the system should calculate and display sensitivity, specificity, and F1-score metrics.
**Validates: Requirements 12.2**

**Property 42: Processing Time Tracking**
*For any* batch of diagnostic cases, the system should track average processing time and flag performance degradation when times exceed baseline by more than 50%.
**Validates: Requirements 12.3**

**Property 43: Clinician Feedback Mechanism**
*For any* diagnostic prediction, the system should provide a mechanism for clinicians to mark it as correct or incorrect.
**Validates: Requirements 12.5**

**Property 44: Feedback Persistence**
*For any* clinician feedback provided, the Local_Database should store this information linked to the corresponding diagnostic result.
**Validates: Requirements 12.6**

## Error Handling

The system implements comprehensive error handling across all components:

### Image Processing Errors

1. **Invalid Format**: Return `InvalidImageError` with supported formats list
2. **Corrupted File**: Log error, skip file in batch processing, notify user
3. **Insufficient Resolution**: Return `ResolutionError` with minimum requirements
4. **DICOM Parse Failure**: Attempt fallback to pixel data only, log metadata extraction failure

### Model Inference Errors

1. **Model Load Failure**: Retain previous model, log error, notify administrator
2. **Out of Memory**: Reduce batch size, trigger garbage collection, retry
3. **Invalid Feature Vector**: Log feature extraction output, return error to user
4. **Prediction Timeout**: Cancel inference after 30 seconds, log timeout, suggest system check

### Database Errors

1. **Connection Failure**: Retry with exponential backoff (3 attempts)
2. **Constraint Violation**: Roll back transaction, log violation, return specific error
3. **Disk Full**: Block writes, alert user immediately, suggest cleanup
4. **Corruption Detected**: Attempt recovery from last backup, log corruption details

### Network Errors

1. **Sync Failure**: Pause sync, log error, retry on next connectivity check
2. **Timeout**: Cancel request after 60 seconds, mark as failed, retry later
3. **Authentication Failure**: Clear credentials, prompt for re-authentication
4. **Server Unavailable**: Queue data locally, retry with exponential backoff

### UI Errors

1. **Render Failure**: Log error, display fallback UI with error message
2. **User Input Validation**: Display inline error messages with correction guidance
3. **Navigation Error**: Log error, return to home screen
4. **Export Failure**: Log error, suggest alternative export format

### Error Recovery Strategies

1. **Graceful Degradation**: Continue operation with reduced functionality when possible
2. **State Preservation**: Save application state before risky operations
3. **User Communication**: Provide clear, actionable error messages in user's language
4. **Automatic Recovery**: Implement retry logic with exponential backoff
5. **Fallback Mechanisms**: Maintain previous working state when updates fail

## Testing Strategy

The FractalLens system requires comprehensive testing to ensure diagnostic accuracy and reliability in rural healthcare settings. We will implement both unit testing and property-based testing as complementary approaches.

### Testing Approach

**Unit Tests**: Focus on specific examples, edge cases, and integration points
- Specific medical image examples with known features
- Edge cases (minimum resolution, maximum file size, corrupted data)
- Error conditions and recovery mechanisms
- Integration between components

**Property-Based Tests**: Verify universal properties across all inputs
- Universal correctness properties that should hold for any valid input
- Comprehensive input coverage through randomization
- Minimum 100 iterations per property test to ensure statistical confidence

Both testing approaches are necessary for comprehensive coverage. Unit tests catch concrete bugs and validate specific scenarios, while property tests verify general correctness across the input space.

### Property-Based Testing Configuration

We will use **Hypothesis** (Python's property-based testing library) for implementing property tests. Each property test will:
- Run a minimum of 100 iterations with randomly generated inputs
- Reference its corresponding design document property via comment tag
- Tag format: `# Feature: fractallens, Property {number}: {property_text}`

Example:
```python
from hypothesis import given, strategies as st

@given(st.images())  # Generate random medical images
def test_complete_feature_extraction(image):
    # Feature: fractallens, Property 1: Complete Feature Extraction
    features = extractor.extract_features(image)
    assert features.fractal_dimension is not None
    assert features.lacunarity is not None
    assert features.regional_features is not None
```

### Test Coverage by Component

**Physics Feature Extractor**:
- Unit tests: Known images with calculated FD/Lacunarity values
- Property tests: Properties 1, 2, 3
- Edge cases: Minimum resolution (256x256), highly noisy images, uniform images

**PINN Model**:
- Unit tests: Specific feature vectors with expected classifications
- Property tests: Properties 4, 5, 6
- Edge cases: Boundary confidence values (0.69, 0.70, 0.71)

**Explainability Generator**:
- Unit tests: Specific heatmap generation examples
- Property tests: Properties 7, 8, 9
- Edge cases: Images with no high-entropy regions, images with all high-entropy

**Local Database**:
- Unit tests: CRUD operations, schema validation
- Property tests: Properties 10, 11, 12
- Edge cases: Concurrent access, large datasets, disk space limits

**Sync Manager**:
- Unit tests: Mock network scenarios (success, failure, timeout)
- Property tests: Properties 13, 14, 15, 16, 17, 18
- Edge cases: Network interruption mid-sync, conflicting timestamps

**Image Preprocessor**:
- Unit tests: Each supported format (DICOM, PNG, JPEG)
- Property tests: Properties 19, 20, 21, 22, 23
- Edge cases: Minimum resolution, maximum file size, corrupted headers

**Explainability Dashboard**:
- Unit tests: UI component rendering, user interactions
- Property tests: Properties 24, 25, 26, 27
- Edge cases: Very large images, slow processing, rapid user interactions

**Model Management**:
- Unit tests: Model loading, version validation
- Property tests: Properties 28, 29, 30
- Edge cases: Incompatible model versions, corrupted model files

**Error Handling**:
- Unit tests: Each error type with specific triggers
- Property tests: Properties 31, 32, 33, 34, 35, 36
- Edge cases: Multiple simultaneous errors, cascading failures

**Security**:
- Unit tests: Authentication flows, encryption validation
- Property tests: Properties 37, 38, 39, 40
- Edge cases: Session edge cases (14:59 vs 15:01 minutes), encryption key rotation

**Performance Monitoring**:
- Unit tests: Metrics calculation with known datasets
- Property tests: Properties 41, 42, 43, 44
- Edge cases: Zero cases, exactly 100 cases, performance degradation detection

### Integration Testing

Beyond unit and property tests, we will implement integration tests for:
1. **End-to-End Diagnostic Flow**: Image upload → processing → results display → database save
2. **Offline-to-Online Transition**: Diagnostic in offline mode → connectivity restored → automatic sync
3. **Model Update Flow**: New model available → validation → loading → diagnostic with new model
4. **Multi-User Scenarios**: Concurrent diagnostics, session management, audit logging

### Test Data

**Synthetic Medical Images**:
- Generate synthetic CT/OCT images with known fractal properties
- Create images with controlled noise levels
- Produce images at various resolutions (256x256 to 2048x2048)

**Real Medical Images** (anonymized):
- Partner with medical institutions for anonymized test datasets
- Include diverse pathology types and imaging conditions
- Maintain ground truth labels for accuracy validation

**Edge Case Images**:
- Minimum resolution (256x256)
- Maximum practical resolution (4096x4096)
- Highly noisy images (SNR < 10dB)
- Uniform images (no texture)
- Corrupted files (truncated, invalid headers)

### Continuous Testing

- Run unit tests on every code commit
- Run property tests nightly (due to longer execution time)
- Run integration tests before each release
- Monitor test coverage (target: >85% code coverage)
- Track property test failure rates to identify flaky tests

### Performance Benchmarking

In addition to functional testing, we will benchmark:
- Feature extraction time per image
- Inference time per image
- Database query performance at various dataset sizes
- Memory usage during batch processing
- Sync throughput under various network conditions

Target performance metrics:
- Feature extraction: < 2 seconds per image
- Inference: < 10 seconds per image on Intel i5 CPU
- Database queries: < 2 seconds for 10,000 patient dataset
- Memory usage: < 4GB RAM during operation
- Model size: < 500MB on disk
