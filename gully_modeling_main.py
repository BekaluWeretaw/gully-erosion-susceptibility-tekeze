# ============================================================================
# COMPLETE GULLY SUSCEPTIBILITY MODELING WORKFLOW
# Belesa Watershed, Northern Ethiopia
# Includes: Data loading, Model training, Ensemble, Validation, Figures, Tables
# ============================================================================

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import warnings
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                           recall_score, f1_score, confusion_matrix, 
                           cohen_kappa_score, log_loss, roc_curve, auc,
                           brier_score_loss, matthews_corrcoef,
                           mean_absolute_error, mean_squared_error,
                           average_precision_score, calibration_curve)
from sklearn.model_selection import train_test_split
import seaborn as sns
warnings.filterwarnings('ignore')

# ============================================================================
# 0. CONFIGURATION - SET YOUR WORKING DIRECTORY HERE
# ============================================================================
WORKING_DIR = r"D:\Manuscript for Bekalu\Chl-a,Water Hyacinth and Gully\Gully Manuscripts Belesa\Gully points\geological_classification\Belsa_GESM"
os.chdir(WORKING_DIR)

print("=" * 80)
print("COMPLETE GULLY SUSCEPTIBILITY MODELING WORKFLOW")
print("Belesa Watershed, Northern Ethiopia")
print("Models: Gradient Boosting, Random Forest, ANN, SVM, Ensemble")
print("=" * 80)

# ============================================================================
# 1. CONFIGURATION AND FILE PATHS
# ============================================================================
TRAINING_FILE = "Gully_Training_Set.csv"
VALIDATION_FILE = "Gully_Validation_Set.csv"
RASTER_FOLDER_NAME = "All_Raster_Data_CF"
SHAPEFILE_PATH = os.path.join(WORKING_DIR, "belsa.shp")
RASTER_FOLDER = os.path.join(WORKING_DIR, RASTER_FOLDER_NAME)
MODELS_DIR = os.path.join(WORKING_DIR, "Trained_Models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Model names
MODEL_NAMES = ['Gradient_Boosting', 'Random_Forest', 'ANN', 'SVM']

# File mapping for conditioning factors
file_mapping = {
    'Drainage_D': 'Re_13_Drainage_Density.tif',
    'TWI': 'Re_12_TWI.tif',
    'Plan_Curva': 'Re_11_Plan_Curvature.tif',
    'Aspect': 'Re_10_Aspect.tif',
    'Elevation': 'Re_09_Elevation.tif',
    'Slope': 'Re_08_Slope.tif',
    'Analytic_H': 'Re_07_Analytic_Hillshade.tif',
    'TRI': 'Re_06_TRI.tif',
    'TPI': 'Re_05_TPI.tif',
    'Convergenc': 'Re_04_Convergence_Index.tif',
    'Profile_Cu': 'Re_03_Profile_Curvature.tif',
    'RSP': 'Re_02_RSP.tif',
    'LS': 'Re_01_LS_Factor.tif',
    'Soil_Organ': 'Re_UTM_M_Belsa_Soil_Organic_Matter.tif',
    'Lithology': 'Re_UTM_M_Belsa_Lithology_Classification.tif',
    'LULC': 'Re_UTM_M_Belsa_Land_Cover_2020.tif',
    'Fault_Dist': 'Re_UTM_M_Belsa_Fault_Distance_km.tif',
    'NDVI': 'Re_UTM_M_Belsa_Central_Gondar_NDVI_2018_CloudMasked.tif',
    'Rainfall': 'Re_UTM_M_Belsa_Annual_Rainfall_2018.tif',
    'Soil_Types': 'Re_UTM_Clay_Soil_Types_Final_Publication_end.tif'
}

# Equal interval thresholds for classification
class_thresholds = np.linspace(0, 1, 6)
class_names = ["Very Low", "Low", "Moderate", "High", "Very High"]

# Create output directories
OUTPUT_DIR = os.path.join(WORKING_DIR, "Gully_Modeling_Results_Complete")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "Figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "Tables")
MAPS_DIR = os.path.join(OUTPUT_DIR, "Susceptibility_Maps")
PROB_DIR = os.path.join(MAPS_DIR, "Probability")
CLASS_DIR = os.path.join(MAPS_DIR, "Classes")
VALIDATION_DIR = os.path.join(OUTPUT_DIR, "Validation_Metrics")

for dir_path in [OUTPUT_DIR, FIGURE_DIR, TABLE_DIR, MAPS_DIR, PROB_DIR, CLASS_DIR, VALIDATION_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# 2. LOAD AND PREPARE DATA
# ============================================================================
print("\n" + "-" * 60)
print("STEP 1: LOADING TRAINING AND VALIDATION DATA")
print("-" * 60)

# Load data
train_data = pd.read_csv(TRAINING_FILE)
val_data = pd.read_csv(VALIDATION_FILE)

# Prepare features and target
X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_val = val_data.drop('Class', axis=1)
y_val = val_data['Class']

print(f"Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Validation: {X_val.shape[0]} samples")
print(f"Gully prevalence: {(y_train.mean()*100):.1f}% (train), {(y_val.mean()*100):.1f}% (val)")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
feature_names = X_train.columns.tolist()

# Save scaler
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

# ============================================================================
# 3. TRAIN MODELS
# ============================================================================
print("\n" + "-" * 60)
print("STEP 2: TRAINING MACHINE LEARNING MODELS")
print("-" * 60)

trained_models = {}

# 3.1 Gradient Boosting
print("\nTraining Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, random_state=42
)
gb_model.fit(X_train_scaled, y_train)
trained_models['Gradient_Boosting'] = gb_model
joblib.dump(gb_model, os.path.join(MODELS_DIR, 'Gradient_Boosting_model.pkl'))
print("  ✓ Gradient Boosting trained and saved")

# 3.2 Random Forest
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
trained_models['Random_Forest'] = rf_model
joblib.dump(rf_model, os.path.join(MODELS_DIR, 'Random_Forest_model.pkl'))
print("  ✓ Random Forest trained and saved")

# 3.3 ANN
print("\nTraining ANN...")
ann_model = MLPClassifier(
    hidden_layer_sizes=(100, 50), activation='relu',
    solver='adam', max_iter=500, random_state=42,
    early_stopping=True, validation_fraction=0.1
)
ann_model.fit(X_train_scaled, y_train)
trained_models['ANN'] = ann_model
joblib.dump(ann_model, os.path.join(MODELS_DIR, 'ANN_model.pkl'))
print("  ✓ ANN trained and saved")

# 3.4 SVM
print("\nTraining SVM...")
svm_model = SVC(
    kernel='rbf', C=1.0, gamma='scale',
    probability=True, random_state=42, max_iter=1000
)
svm_model.fit(X_train_scaled, y_train)
trained_models['SVM'] = svm_model
joblib.dump(svm_model, os.path.join(MODELS_DIR, 'SVM_model.pkl'))
print("  ✓ SVM trained and saved")

# ============================================================================
# 4. CREATE ENSEMBLE MODEL
# ============================================================================
print("\n" + "-" * 60)
print("STEP 3: CREATING ENSEMBLE MODEL")
print("-" * 60)

class EnsembleGullyModel:
    def __init__(self, base_models=None, weights=None, name="Ensemble"):
        self.base_models = base_models if base_models is not None else {}
        self.weights = weights
        self.name = name
        self.n_models = len(self.base_models)
        
        if self.weights is None and self.n_models > 0:
            self.weights = np.ones(self.n_models) / self.n_models
    
    def predict_proba(self, X):
        all_probs = []
        valid_weights = []
        
        for i, (name, model) in enumerate(self.base_models.items()):
            try:
                probs = model.predict_proba(X)
                if probs.ndim > 1:
                    probs = probs[:, 1]
                all_probs.append(probs)
                valid_weights.append(self.weights[i])
            except:
                continue
        
        valid_weights = np.array(valid_weights) / np.sum(valid_weights)
        weighted_probs = np.average(all_probs, axis=0, weights=valid_weights)
        return weighted_probs
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# Calculate weights based on validation AUC
auc_scores = []
for name, model in trained_models.items():
    y_proba = model.predict_proba(X_val_scaled)[:, 1]
    auc_val = roc_auc_score(y_val, y_proba)
    auc_scores.append(auc_val)
    print(f"  {name}: AUC = {auc_val:.4f}")

weights = np.array(auc_scores) / np.sum(auc_scores)
ensemble = EnsembleGullyModel(trained_models, weights, "Ensemble")
trained_models['Ensemble'] = ensemble
print(f"  ✓ Ensemble created with weights: {dict(zip(trained_models.keys(), np.round(weights, 4)))}")

# ============================================================================
# 5. COMPREHENSIVE VALIDATION METRICS
# ============================================================================
print("\n" + "-" * 60)
print("STEP 4: CALCULATING VALIDATION METRICS")
print("-" * 60)

all_metrics = []
roc_data = {}
calibration_data = {}

for name, model in trained_models.items():
    # Get predictions
    if name == 'Ensemble':
        y_proba = model.predict_proba(X_val_scaled)
    else:
        y_proba = model.predict_proba(X_val_scaled)[:, 1]
    y_pred = model.predict(X_val_scaled)
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    metrics = {
        'Model': name,
        'AUC': roc_auc_score(y_val, y_proba),
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred, zero_division=0),
        'Recall': recall_score(y_val, y_pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'F1_Score': f1_score(y_val, y_pred, zero_division=0),
        'Kappa': cohen_kappa_score(y_val, y_pred),
        'Log_Loss': log_loss(y_val, y_proba),
        'Brier_Score': brier_score_loss(y_val, y_proba),
        'MCC': matthews_corrcoef(y_val, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_val, y_proba)),
        'MAE': mean_absolute_error(y_val, y_proba),
        'Avg_Precision': average_precision_score(y_val, y_proba),
        'Youden_Index': (tp/(tp+fn)) + (tn/(tn+fp)) - 1,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    }
    all_metrics.append(metrics)
    
    # ROC data
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': metrics['AUC']}
    
    # Calibration data
    fraction_pos, mean_pred = calibration_curve(y_val, y_proba, n_bins=10)
    calibration_data[name] = {
        'fraction_pos': fraction_pos,
        'mean_pred': mean_pred,
        'log_loss': metrics['Log_Loss'],
        'brier': metrics['Brier_Score']
    }
    
    print(f"  ✓ {name}: AUC={metrics['AUC']:.4f}, LogLoss={metrics['Log_Loss']:.4f}")

metrics_df = pd.DataFrame(all_metrics)
metrics_df = metrics_df.sort_values('AUC', ascending=False)
metrics_df.to_csv(os.path.join(VALIDATION_DIR, '01_all_metrics.csv'), index=False)

print("\n" + "=" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 80)
print(metrics_df[['Model', 'AUC', 'Accuracy', 'F1_Score', 'Kappa', 'Log_Loss']].to_string(index=False))

# ============================================================================
# 6. LOAD SHAPEFILE
# ============================================================================
print("\n" + "-" * 60)
print("STEP 5: LOADING STUDY AREA SHAPEFILE")
print("-" * 60)

def load_shapefile(shapefile_path):
    try:
        gdf = gpd.read_file(shapefile_path)
        print(f"✓ Shapefile loaded: {os.path.basename(shapefile_path)}")
        if len(gdf) > 1:
            combined_geom = gdf.geometry.unary_union
        else:
            combined_geom = gdf.geometry.iloc[0]
        return gdf, combined_geom
    except:
        return None, None

study_area_gdf, study_area_geom = load_shapefile(SHAPEFILE_PATH)
use_shapefile = study_area_geom is not None

# ============================================================================
# 7. LOAD RASTER DATA
# ============================================================================
print("\n" + "-" * 60)
print("STEP 6: LOADING RASTER DATA")
print("-" * 60)

def clip_raster_with_shapefile(raster_path, shape_geom, shape_gdf):
    try:
        with rasterio.open(raster_path) as src:
            if shape_gdf is not None and shape_gdf.crs != src.crs:
                shape_gdf_reproj = shape_gdf.to_crs(src.crs)
                shape_geom_reproj = shape_gdf_reproj.geometry.unary_union
            else:
                shape_geom_reproj = shape_geom
            
            out_image, out_transform = mask(src, [mapping(shape_geom_reproj)], crop=True, filled=True)
            data = out_image[0].astype(np.float32)
            
            nodata = src.nodata if src.nodata is not None else -9999
            data[data == nodata] = np.nan
            
            return data, src.profile
    except Exception as e:
        print(f"  Error clipping: {e}")
        return None, None

# Find available rasters
available_features = []
for feature in feature_names:
    if feature in file_mapping:
        path = os.path.join(RASTER_FOLDER, file_mapping[feature])
        if os.path.exists(path):
            available_features.append((feature, path))

print(f"Found {len(available_features)} rasters")

# Load first raster for reference
with rasterio.open(available_features[0][1]) as src:
    if use_shapefile and study_area_gdf is not None:
        out_image, out_transform = mask(src, [mapping(study_area_geom)], crop=True)
        first_data = out_image[0]
        profile = src.profile
        profile.update({'height': first_data.shape[0], 'width': first_data.shape[1],
                       'transform': out_transform})
    else:
        first_data = src.read(1)
        profile = src.profile

height, width = first_data.shape
pixel_size_x = abs(profile['transform'][0])
pixel_size_y = abs(profile['transform'][4])
pixel_area_ha = (pixel_size_x * pixel_size_y) / 10000

print(f"Study area: {height} × {width} pixels")
print(f"Pixel size: {pixel_size_x:.1f} × {pixel_size_y:.1f} m")
print(f"Total area: {height * width * pixel_area_ha:,.1f} hectares")

# Load all rasters
raster_stack = np.zeros((len(feature_names), height, width), dtype=np.float32)

for i, feature in enumerate(feature_names):
    matching = [f for f, path in available_features if f == feature]
    if matching:
        path = [p for f, p in available_features if f == feature][0]
        with rasterio.open(path) as src:
            if use_shapefile and study_area_geom is not None:
                data, _ = mask(src, [mapping(study_area_geom)], crop=True)
                data = data[0]
            else:
                data = src.read(1)
        
        if data.shape != (height, width):
            data = zoom(data, (height/data.shape[0], width/data.shape[1]), order=1)
        
        data = data.astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        data = np.nan_to_num(data, nan=X_train[feature].mean())
        data = np.clip(data, X_train[feature].mean() - 5*X_train[feature].std(), 
                      X_train[feature].mean() + 5*X_train[feature].std())
        
        raster_stack[i] = data
        print(f"  ✓ {i+1:2d}. {feature}")
    else:
        raster_stack[i] = X_train[feature].mean()
        print(f"  ⚠ {i+1:2d}. {feature} (filled with mean)")

# Prepare for prediction
X_watershed = raster_stack.reshape(len(feature_names), -1).T
X_watershed_scaled = scaler.transform(X_watershed)
print(f"Ready to predict: {X_watershed.shape[0]:,} pixels")

# ============================================================================
# 8. GENERATE SUSCEPTIBILITY MAPS
# ============================================================================
print("\n" + "-" * 60)
print("STEP 7: GENERATING SUSCEPTIBILITY MAPS")
print("-" * 60)

output_profile = profile.copy()
output_profile.update({'dtype': 'float32', 'count': 1, 'compress': 'lzw'})

def classify_equal_interval(map_data, thresholds):
    classes = np.zeros_like(map_data, dtype=np.int16)
    for i in range(5):
        if i == 0:
            mask = (map_data >= thresholds[i]) & (map_data < thresholds[i+1])
        elif i == 4:
            mask = (map_data >= thresholds[i]) & (map_data <= thresholds[i+1])
        else:
            mask = (map_data >= thresholds[i]) & (map_data < thresholds[i+1])
        classes[mask] = i + 1
    return classes

area_stats = []

for name, model in trained_models.items():
    print(f"\nProcessing {name}...")
    
    # Predict in batches
    n_samples = X_watershed_scaled.shape[0]
    predictions = np.zeros(n_samples, dtype=np.float32)
    batch_size = 500000
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = X_watershed_scaled[start:end]
        
        if name == 'Ensemble':
            pred = model.predict_proba(batch)
        else:
            pred = model.predict_proba(batch)[:, 1]
        
        predictions[start:end] = pred
    
    sus_map = predictions.reshape(height, width)
    
    # Save probability map
    with rasterio.open(os.path.join(PROB_DIR, f'{name}_Probability.tif'), 'w', **output_profile) as dst:
        dst.write(sus_map.astype(np.float32), 1)
    
    # Classify and save
    classes = classify_equal_interval(sus_map, class_thresholds)
    class_profile = output_profile.copy()
    class_profile.update({'dtype': 'int16'})
    
    with rasterio.open(os.path.join(CLASS_DIR, f'{name}_Classes.tif'), 'w', **class_profile) as dst:
        dst.write(classes.astype(np.int16), 1)
    
    # Calculate statistics
    stats = {'Model': name}
    total_pixels = height * width
    
    for i in range(1, 6):
        count = (classes == i).sum()
        area = count * pixel_area_ha
        percent = count / total_pixels * 100
        stats[f'Class_{i}_ha'] = area
        stats[f'Class_{i}_pct'] = percent
    
    stats['High_Risk_ha'] = stats['Class_4_ha'] + stats['Class_5_ha']
    stats['High_Risk_pct'] = stats['Class_4_pct'] + stats['Class_5_pct']
    area_stats.append(stats)
    
    print(f"  High Risk Area: {stats['High_Risk_ha']:,.0f} ha ({stats['High_Risk_pct']:.1f}%)")

area_df = pd.DataFrame(area_stats)
area_df.to_csv(os.path.join(VALIDATION_DIR, '02_area_statistics.csv'), index=False)

# ============================================================================
# 9. GENERATE FIGURE 5 - ROC CURVES
# ============================================================================
print("\n" + "-" * 60)
print("STEP 8: GENERATING FIGURE 5 - ROC CURVES")
print("-" * 60)

plt.style.use('default')
fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor('white')

colors = {
    'Gradient_Boosting': '#FFE194', 'Random_Forest': '#45B7D1',
    'ANN': '#FF6B6B', 'SVM': '#4ECDC4', 'Ensemble': '#9B59B6'
}
line_styles = ['-', '--', '-.', ':', (0, (5, 2))]

for idx, (name, data) in enumerate(roc_data.items()):
    color = colors.get(name, f'C{idx}')
    style = line_styles[idx % len(line_styles)]
    lw = 3.5 if name == 'Ensemble' else 2.5
    ax.plot(data['fpr'], data['tpr'], color=color, linestyle=style,
            linewidth=lw, label=f"{name} (AUC = {data['auc']:.3f})")

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, 
        label='Random Classifier (AUC = 0.5)')

ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
ax.set_title('Receiver Operating Characteristic (ROC) Curves\nGully Susceptibility Models - Belesa Watershed', 
            fontsize=16, fontweight='bold', pad=20)

legend = ax.legend(loc='lower right', fontsize=12, framealpha=0.95,
                  fancybox=True, shadow=True, borderpad=1)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('black')

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.text(0.98, 0.02, 'FIGURE 5', transform=ax.transAxes, fontsize=16,
        fontweight='bold', ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8, edgecolor='black'))

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'Figure5_ROC_Curves.png'), dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(FIGURE_DIR, 'Figure5_ROC_Curves.pdf'), dpi=600, bbox_inches='tight', format='pdf')
plt.savefig(os.path.join(FIGURE_DIR, 'Figure5_ROC_Curves.svg'), dpi=600, bbox_inches='tight', format='svg')
plt.close()
print("  ✓ Figure 5 saved")

# ============================================================================
# 10. GENERATE FIGURE 6 - LOG LOSS CALIBRATION
# ============================================================================
print("\n" + "-" * 60)
print("STEP 9: GENERATING FIGURE 6 - LOG LOSS CALIBRATION")
print("-" * 60)

model_mapping = {'ANN': 'A', 'SVM': 'B', 'Random_Forest': 'C', 'Gradient_Boosting': 'D'}
model_titles = {
    'ANN': 'Artificial Neural Network', 'SVM': 'Support Vector Machine',
    'Random_Forest': 'Random Forest', 'Gradient_Boosting': 'Gradient Boosting'
}
model_colors = {'ANN': '#FF6B6B', 'SVM': '#4ECDC4', 'Random_Forest': '#45B7D1', 'Gradient_Boosting': '#FFE194'}

fig = plt.figure(figsize=(14, 12))
fig.patch.set_facecolor('white')
fig.suptitle('Performance Validation of Machine Learning Models for Gully Erosion\nSusceptibility Modeling Using Log Loss',
            fontsize=14, fontweight='bold', y=0.98)

gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

for idx, (model_name, panel) in enumerate(model_mapping.items()):
    ax = fig.add_subplot(gs[idx])
    ax.set_facecolor('#f8f9fa')
    data = calibration_data[model_name]
    
    ax.plot(data['mean_pred'], data['fraction_pos'], 'o-', color='#2E86AB', 
            linewidth=2, markersize=6, markerfacecolor='white', 
            markeredgewidth=1.5, markeredgecolor='#2E86AB')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    
    ax.text(0.5, 0.95, f'Log Loss = {data["log_loss"]:.4f}', transform=ax.transAxes,
            fontsize=11, fontweight='bold', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax.set_xlabel('Predicted Probability', fontsize=11)
    ax.set_ylabel('Observed Probability', fontsize=11)
    ax.set_title(f'({panel}) {model_titles[model_name]}', fontsize=12, fontweight='bold', loc='left')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

fig.text(0.5, 0.02, 'FIGURE 6', fontsize=14, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8, edgecolor='black'))

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.05)
plt.savefig(os.path.join(FIGURE_DIR, 'Figure6_LogLoss_Calibration.png'), dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(FIGURE_DIR, 'Figure6_LogLoss_Calibration.pdf'), dpi=600, bbox_inches='tight', format='pdf')
plt.savefig(os.path.join(FIGURE_DIR, 'Figure6_LogLoss_Calibration.svg'), dpi=600, bbox_inches='tight', format='svg')
plt.close()
print("  ✓ Figure 6 saved")

# ============================================================================
# 11. GENERATE FEATURE IMPORTANCE FIGURE
# ============================================================================
print("\n" + "-" * 60)
print("STEP 10: GENERATING FEATURE IMPORTANCE FIGURE")
print("-" * 60)

short_names = {
    'RSP': 'RSP', 'NDVI': 'NDVI', 'Soil_Organ': 'SOM', 'Drainage_D': 'DD',
    'Rainfall': 'RF', 'Elevation': 'ELV', 'Fault_Dist': 'FD', 'TWI': 'TWI',
    'Slope': 'SLP', 'LS': 'LS', 'TPI': 'TPI', 'TRI': 'TRI', 'Aspect': 'ASP',
    'Plan_Curva': 'PLC', 'Profile_Cu': 'PRC', 'Lithology': 'LIT', 'LULC': 'LULC',
    'Analytic_H': 'AH', 'Convergenc': 'CI', 'Soil_Types': 'ST'
}

models_2x2 = ['Gradient_Boosting', 'Random_Forest', 'ANN', 'SVM']
model_titles_fi = ['(a) Gradient Boosting', '(b) Random Forest', '(c) ANN', '(d) SVM']

fig = plt.figure(figsize=(16, 14))
fig.suptitle('Feature Importance for Gully Susceptibility Models\nBelesa Watershed, Northern Ethiopia', 
             fontsize=18, fontweight='bold', y=0.98)

gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

for idx, model_name in enumerate(models_2x2):
    if model_name in trained_models:
        ax = fig.add_subplot(gs[idx])
        model = trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_ * 100
        else:
            continue
            
        sorted_idx = np.argsort(imp)[-15:]
        sorted_imp = imp[sorted_idx]
        sorted_features = [short_names.get(feature_names[i], feature_names[i][:4]) for i in sorted_idx]
        
        y_pos = np.arange(len(sorted_features))
        bars = ax.barh(y_pos, sorted_imp, color=model_colors.get(model_name, '#2E86AB'), 
                       edgecolor='black', linewidth=0.5, alpha=0.8, height=0.6)
        
        for i, (bar, val) in enumerate(zip(bars, sorted_imp)):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f}%', 
                   va='center', fontsize=8, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features, fontsize=10)
        ax.set_xlabel('Relative Importance (%)', fontsize=11)
        ax.set_title(model_titles_fi[idx], fontsize=14, fontweight='bold', color=model_colors.get(model_name, '#2E86AB'))
        ax.set_xlim(0, 105)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(os.path.join(FIGURE_DIR, 'Figure_Feature_Importance_2x2.png'), dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(FIGURE_DIR, 'Figure_Feature_Importance_2x2.pdf'), dpi=600, bbox_inches='tight', format='pdf')
plt.savefig(os.path.join(FIGURE_DIR, 'Figure_Feature_Importance_2x2.svg'), dpi=600, bbox_inches='tight', format='svg')
plt.close()
print("  ✓ Feature Importance figure saved")

# ============================================================================
# 12. SAVE ALL TABLES
# ============================================================================
print("\n" + "-" * 60)
print("STEP 11: SAVING VALIDATION TABLES")
print("-" * 60)

# Table 1: Performance Metrics
table1 = metrics_df[['Model', 'AUC', 'Accuracy', 'Precision', 'Recall', 
                     'Specificity', 'F1_Score', 'Kappa', 'Log_Loss', 'Brier_Score']].round(4)
table1.to_csv(os.path.join(TABLE_DIR, 'Table1_Performance_Metrics.csv'), index=False)
print("  ✓ Table 1 saved")

# Table 2: Area Statistics
table2 = area_df[['Model', 'Class_1_pct', 'Class_2_pct', 'Class_3_pct', 
                  'Class_4_pct', 'Class_5_pct', 'High_Risk_pct']].round(2)
table2.to_csv(os.path.join(TABLE_DIR, 'Table2_Area_Distribution.csv'), index=False)
print("  ✓ Table 2 saved")

# Table 3: Confusion Matrix
table3 = metrics_df[['Model', 'TP', 'TN', 'FP', 'FN']].astype({'TP': int, 'TN': int, 'FP': int, 'FN': int})
table3.to_csv(os.path.join(TABLE_DIR, 'Table3_Confusion_Matrix.csv'), index=False)
print("  ✓ Table 3 saved")

# Comprehensive Table
comprehensive = pd.merge(
    metrics_df[['Model', 'AUC', 'Accuracy', 'F1_Score', 'Kappa', 'Log_Loss']],
    area_df[['Model', 'High_Risk_pct']], on='Model'
).sort_values('AUC', ascending=False).round(4)
comprehensive.to_csv(os.path.join(TABLE_DIR, 'Table4_Comprehensive_Comparison.csv'), index=False)
print("  ✓ Table 4 saved")

# LaTeX Table
latex_table = """\\begin{table}[h]
\\centering
\\caption{Performance validation metrics of machine learning models for gully erosion susceptibility modeling}
\\label{tab:model_performance}
\\begin{tabular}{|l|c|c|c|c|c|c|}
\\hline
\\textbf{Model} & \\textbf{AUC} & \\textbf{Accuracy} & \\textbf{F1-Score} & \\textbf{Kappa} & \\textbf{Log Loss} & \\textbf{High Risk (\\%)} \\\\
\\hline
"""

for _, row in comprehensive.iterrows():
    latex_table += f"{row['Model']} & {row['AUC']:.4f} & {row['Accuracy']:.4f} & "
    latex_table += f"{row['F1_Score']:.4f} & {row['Kappa']:.4f} & {row['Log_Loss']:.4f} & "
    latex_table += f"{row['High_Risk_pct']:.2f} \\\\\n\\hline\n"

latex_table += """\\end{tabular}
\\end{table}"""

with open(os.path.join(TABLE_DIR, 'LaTeX_Table.txt'), 'w') as f:
    f.write(latex_table)
print("  ✓ LaTeX table saved")

# ============================================================================
# 13. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY - ALL FILES GENERATED")
print("=" * 80)

best_model = comprehensive.iloc[0]['Model']
best_auc = comprehensive.iloc[0]['AUC']

print(f"""
🏆 BEST MODEL: {best_model} (AUC = {best_auc:.4f})

📁 OUTPUT DIRECTORY: {OUTPUT_DIR}

📊 FIGURES:
   {FIGURE_DIR}/
   ├── Figure5_ROC_Curves.png/pdf/svg
   ├── Figure6_LogLoss_Calibration.png/pdf/svg
   └── Figure_Feature_Importance_2x2.png/pdf/svg

📋 TABLES:
   {TABLE_DIR}/
   ├── Table1_Performance_Metrics.csv
   ├── Table2_Area_Distribution.csv
   ├── Table3_Confusion_Matrix.csv
   ├── Table4_Comprehensive_Comparison.csv
   └── LaTeX_Table.txt

🗺️ SUSCEPTIBILITY MAPS:
   {MAPS_DIR}/
   ├── Probability/ - Probability maps for all models
   └── Classes/ - Classified maps for all models

✅ All files ready for publication!
""")

print("=" * 80)
print("✅ COMPLETE WORKFLOW FINISHED SUCCESSFULLY!")
print("=" * 80)
print(f"\nFinished at: {pd.Timestamp.now()}")