"""
Complete Transfer Learning Pipeline for Fish Classification
Trains all major pre-trained models with proper regularization
"""

import os
import argparse
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNet, MobileNetV2, 
    InceptionV3, EfficientNetB0, EfficientNetB1,
    DenseNet121, Xception
)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class TransferLearningTrainer:
    def __init__(self, data_dir, output_dir="models_output", img_size=(224, 224), batch_size=16):
        """Initialize trainer with all transfer learning models"""
        self.data_root = Path(data_dir)
        self.train_dir = self.data_root / "train"
        self.val_dir = self.data_root / "val"
        self.test_dir = self.data_root / "test"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        
        self.img_size = img_size
        self.batch_size = batch_size
        
        self.num_classes = None
        self.class_names = None
        self.class_weights = None
        
        self.history_dict = {}
        self.models_dict = {}
        self.results_dict = {}
        
        # Available models configuration
        self.available_models = {
            'MobileNet': {'model': MobileNet, 'size': (224, 224), 'preprocess': tf.keras.applications.mobilenet.preprocess_input},
            'MobileNetV2': {'model': MobileNetV2, 'size': (224, 224), 'preprocess': tf.keras.applications.mobilenet_v2.preprocess_input},
            'VGG16': {'model': VGG16, 'size': (224, 224), 'preprocess': tf.keras.applications.vgg16.preprocess_input},
            'ResNet50': {'model': ResNet50, 'size': (224, 224), 'preprocess': tf.keras.applications.resnet50.preprocess_input},
            'InceptionV3': {'model': InceptionV3, 'size': (299, 299), 'preprocess': tf.keras.applications.inception_v3.preprocess_input},
            'EfficientNetB0': {'model': EfficientNetB0, 'size': (224, 224), 'preprocess': None},
            'EfficientNetB1': {'model': EfficientNetB1, 'size': (240, 240), 'preprocess': None},
            'DenseNet121': {'model': DenseNet121, 'size': (224, 224), 'preprocess': tf.keras.applications.densenet.preprocess_input},
            'Xception': {'model': Xception, 'size': (299, 299), 'preprocess': tf.keras.applications.xception.preprocess_input}
        }
        
        print("="*70)
        print("🚀 TRANSFER LEARNING TRAINER INITIALIZED")
        print("="*70)
        print(f"📁 Dataset: {self.data_root}")
        print(f"💾 Output: {self.output_dir}")
        print(f"📦 Batch size: {self.batch_size}")
        
        self.validate_dataset()
    
    def validate_dataset(self):
        """Validate dataset structure"""
        if not self.train_dir.exists() or not self.val_dir.exists():
            raise ValueError(f"Missing train or val directory in {self.data_root}")
        
        # Count images
        def count_images(folder):
            total = 0
            per_class = {}
            for cls_dir in folder.iterdir():
                if cls_dir.is_dir():
                    imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
                    per_class[cls_dir.name] = len(imgs)
                    total += len(imgs)
            return total, per_class
        
        train_total, train_counts = count_images(self.train_dir)
        val_total, val_counts = count_images(self.val_dir)
        
        print(f"\n📊 Dataset Info:")
        print(f"  • Train: {train_total} images, {len(train_counts)} classes")
        print(f"  • Val: {val_total} images, {len(val_counts)} classes")
        
        if train_total < 100:
            print("\n⚠️  WARNING: Very small dataset! Expect challenges with deep learning.")
    
    def prepare_data(self, model_name):
        """Prepare data generators with model-specific image size"""
        model_config = self.available_models[model_name]
        img_size = model_config['size']
        
        print(f"\n📊 Preparing data for {model_name} (size: {img_size})...")
        
        # Strong augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1.0/255,
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode="reflect"
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator(rescale=1.0/255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False
        )
        
        # Set class info
        self.num_classes = len(train_generator.class_indices)
        self.class_names = list(train_generator.class_indices.keys())
        
        # Calculate class weights
        counter = Counter(train_generator.classes)
        max_val = float(max(counter.values()))
        self.class_weights = {cls: max_val/n for cls, n in counter.items()}
        
        # Save class indices (only once)
        if not (self.output_dir / "class_indices.json").exists():
            with open(self.output_dir / "class_indices.json", "w") as f:
                json.dump(train_generator.class_indices, f, indent=2)
        
        return train_generator, val_generator
    
    def build_model(self, model_name, fine_tune_layers=0):
        """Build transfer learning model with specified architecture"""
        print(f"\n🔨 Building {model_name}...")
        
        model_config = self.available_models[model_name]
        base_model_class = model_config['model']
        img_size = model_config['size']
        
        # Create base model
        base_model = base_model_class(
            weights='imagenet',
            include_top=False,
            input_shape=(*img_size, 3)
        )
        
        # Freeze base layers initially
        base_model.trainable = False
        
        # If fine-tuning, unfreeze top layers
        if fine_tune_layers > 0:
            base_model.trainable = True
            # Freeze all but the last fine_tune_layers
            for layer in base_model.layers[:-fine_tune_layers]:
                layer.trainable = False
            print(f"  • Fine-tuning last {fine_tune_layers} layers")
        else:
            print(f"  • All base layers frozen")
        
        # Build complete model with regularization
        inputs = layers.Input(shape=(*img_size, 3))
        
        # Add preprocessing if available
        x = inputs
        if model_config['preprocess'] is not None:
            x = model_config['preprocess'](x)
        
        # Add noise for regularization
        x = layers.GaussianNoise(0.1)(x)
        
        # Base model
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Classification head with regularization
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax',
                              kernel_regularizer=regularizers.l2(0.01))(x)
        
        model = models.Model(inputs, outputs)
        
        # Count parameters
        total_params = model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        
        print(f"  • Total parameters: {total_params:,}")
        print(f"  • Trainable parameters: {trainable_params:,}")
        
        return model, base_model
    
    def compile_model(self, model, learning_rate=0.001):
        """Compile model with optimizer"""
        model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        return model
    
    def get_callbacks(self, model_name, patience=7):
        """Get training callbacks"""
        return [
            callbacks.ModelCheckpoint(
                filepath=str(self.output_dir / f"{model_name}_best.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.CSVLogger(
                str(self.output_dir / f"{model_name}_training_log.csv")
            )
        ]
    
    def train_model(self, model_name, epochs=30, fine_tune_epochs=10):
        """Complete training pipeline for a model"""
        print(f"\n{'='*70}")
        print(f"🚀 TRAINING {model_name.upper()}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Prepare data
        train_gen, val_gen = self.prepare_data(model_name)
        
        # Build model (frozen base)
        model, base_model = self.build_model(model_name, fine_tune_layers=0)
        
        # Stage 1: Train with frozen base
        print(f"\n📌 Stage 1: Training with frozen base ({epochs} epochs)")
        print("-"*50)
        
        model = self.compile_model(model, learning_rate=0.001)
        
        history_frozen = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            class_weight=self.class_weights,
            callbacks=self.get_callbacks(f"{model_name}_frozen"),
            verbose=1
        )
        
        # Evaluate frozen model
        frozen_metrics = self.evaluate_model(model, val_gen, f"{model_name}_frozen")
        
        # Stage 2: Fine-tuning (optional, only if frozen performance is good)
        history_finetuned = None
        if frozen_metrics['accuracy'] > 0.5 and fine_tune_epochs > 0:
            print(f"\n📌 Stage 2: Fine-tuning ({fine_tune_epochs} epochs)")
            print("-"*50)
            
            # Rebuild model with unfrozen layers
            model, _ = self.build_model(model_name, fine_tune_layers=20)
            model = self.compile_model(model, learning_rate=0.0001)  # Lower LR for fine-tuning
            
            history_finetuned = model.fit(
                train_gen,
                epochs=fine_tune_epochs,
                validation_data=val_gen,
                class_weight=self.class_weights,
                callbacks=self.get_callbacks(f"{model_name}_finetuned"),
                verbose=1
            )
            
            # Evaluate fine-tuned model
            finetuned_metrics = self.evaluate_model(model, val_gen, f"{model_name}_finetuned")
        
        # Save final model
        model.save(self.output_dir / f"{model_name}_final.h5")
        
        # Store results
        self.models_dict[model_name] = model
        self.history_dict[model_name] = {
            'frozen': history_frozen.history,
            'finetuned': history_finetuned.history if history_finetuned else None
        }
        
        # Plot training curves
        self.plot_training_history(model_name)
        
        # Training summary
        elapsed_time = (time.time() - start_time) / 60
        print(f"\n✅ {model_name} training completed in {elapsed_time:.2f} minutes")
        
        return model
    
    def evaluate_model(self, model, val_generator, model_name):
        """Evaluate model performance"""
        print(f"\n📊 Evaluating {model_name}...")
        
        # Get metrics
        metrics = model.evaluate(val_generator, verbose=0)
        
        # Get predictions for detailed metrics
        predictions = model.predict(val_generator, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = val_generator.classes
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Store results
        result = {
            'accuracy': metrics[1],
            'precision': metrics[2] if len(metrics) > 2 else 0,
            'recall': metrics[3] if len(metrics) > 3 else 0,
            'f1_score': report['weighted avg']['f1-score'],
            'loss': metrics[0],
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        self.results_dict[model_name] = result
        
        print(f"  • Accuracy: {result['accuracy']:.4f}")
        print(f"  • Precision: {result['precision']:.4f}")
        print(f"  • Recall: {result['recall']:.4f}")
        print(f"  • F1-Score: {result['f1_score']:.4f}")
        print(f"  • Loss: {result['loss']:.4f}")
        
        return result
    
    def plot_training_history(self, model_name):
        """Plot training curves"""
        history = self.history_dict[model_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Frozen model curves
        if history['frozen']:
            frozen_hist = history['frozen']
            epochs_frozen = range(1, len(frozen_hist['accuracy']) + 1)
            
            # Accuracy
            axes[0, 0].plot(epochs_frozen, frozen_hist['accuracy'], 'b-', label='Train', linewidth=2)
            axes[0, 0].plot(epochs_frozen, frozen_hist['val_accuracy'], 'r-', label='Val', linewidth=2)
            axes[0, 0].set_title(f'{model_name} - Frozen Base Accuracy')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Loss
            axes[0, 1].plot(epochs_frozen, frozen_hist['loss'], 'b-', label='Train', linewidth=2)
            axes[0, 1].plot(epochs_frozen, frozen_hist['val_loss'], 'r-', label='Val', linewidth=2)
            axes[0, 1].set_title(f'{model_name} - Frozen Base Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Fine-tuned model curves
        if history['finetuned']:
            ft_hist = history['finetuned']
            epochs_ft = range(1, len(ft_hist['accuracy']) + 1)
            
            # Accuracy
            axes[1, 0].plot(epochs_ft, ft_hist['accuracy'], 'g-', label='Train', linewidth=2)
            axes[1, 0].plot(epochs_ft, ft_hist['val_accuracy'], 'orange', label='Val', linewidth=2)
            axes[1, 0].set_title(f'{model_name} - Fine-tuned Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Loss
            axes[1, 1].plot(epochs_ft, ft_hist['loss'], 'g-', label='Train', linewidth=2)
            axes[1, 1].plot(epochs_ft, ft_hist['val_loss'], 'orange', label='Val', linewidth=2)
            axes[1, 1].set_title(f'{model_name} - Fine-tuned Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Fine-tuning', ha='center', va='center', fontsize=12)
            axes[1, 1].text(0.5, 0.5, 'No Fine-tuning', ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / f'{model_name}_training_curves.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def train_all_models(self, models_list=None, epochs=20, fine_tune_epochs=10):
        """Train all specified transfer learning models"""
        if models_list is None:
            # Default: train the most efficient models
            models_list = ['MobileNet', 'MobileNetV2', 'EfficientNetB0', 'ResNet50', 'DenseNet121']
        
        print("\n" + "="*70)
        print(f"🚀 TRAINING {len(models_list)} TRANSFER LEARNING MODELS")
        print("="*70)
        print(f"Models to train: {', '.join(models_list)}")
        print(f"Epochs per model: {epochs} (frozen) + {fine_tune_epochs} (fine-tuning)")
        
        results_summary = []
        
        for i, model_name in enumerate(models_list, 1):
            print(f"\n[{i}/{len(models_list)}] Processing {model_name}...")
            
            try:
                # Train model
                model = self.train_model(
                    model_name,
                    epochs=epochs,
                    fine_tune_epochs=fine_tune_epochs
                )
                
                # Add to summary
                if model_name in self.results_dict:
                    results_summary.append({
                        'Model': model_name,
                        'Accuracy': self.results_dict[model_name]['accuracy'],
                        'Precision': self.results_dict[model_name]['precision'],
                        'Recall': self.results_dict[model_name]['recall'],
                        'F1-Score': self.results_dict[model_name]['f1_score'],
                        'Loss': self.results_dict[model_name]['loss']
                    })
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
                continue
        
        # Create comparison report
        self.create_comparison_report(results_summary)
        
        return results_summary
    
    def create_comparison_report(self, results_summary):
        """Create comprehensive comparison report"""
        if not results_summary:
            print("No results to compare")
            return
        
        print("\n" + "="*70)
        print("📊 MODEL COMPARISON REPORT")
        print("="*70)
        
        # Convert to DataFrame
        df = pd.DataFrame(results_summary)
        df = df.sort_values('Accuracy', ascending=False)
        
        # Display table
        print("\n📈 Performance Ranking:")
        print("-"*50)
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Save to CSV
        df.to_csv(self.output_dir / 'model_comparison.csv', index=False)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy comparison
        axes[0, 0].bar(df['Model'], df['Accuracy'], color='skyblue', edgecolor='navy')
        axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(df['Accuracy']):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # F1-Score comparison
        axes[0, 1].bar(df['Model'], df['F1-Score'], color='lightgreen', edgecolor='darkgreen')
        axes[0, 1].set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(df['F1-Score']):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Loss comparison
        axes[1, 0].bar(df['Model'], df['Loss'], color='salmon', edgecolor='darkred')
        axes[1, 0].set_title('Model Loss Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(df['Loss']):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Combined metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            offset = (i - 1.5) * width
            axes[1, 1].bar(x + offset, df[metric], width, label=metric)
        
        axes[1, 1].set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(df['Model'], rotation=45)
        axes[1, 1].legend()
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Best model
        best_model = df.iloc[0]
        print(f"\n🏆 BEST MODEL: {best_model['Model']}")
        print(f"   • Accuracy: {best_model['Accuracy']:.4f}")
        print(f"   • F1-Score: {best_model['F1-Score']:.4f}")
        print(f"   • Loss: {best_model['Loss']:.4f}")
        
        # Save results to JSON
        with open(self.output_dir / 'training_results.json', 'w') as f:
            json.dump({
                'summary': results_summary,
                'best_model': best_model.to_dict(),
                'all_results': self.results_dict
            }, f, indent=2)
        
        print(f"\n✅ Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train All Transfer Learning Models')
    parser.add_argument('--data', type=str,
                       default=r"D:\Machine Learning\Fish Classification\Dataset\images\data",
                       help='Dataset path with train/val folders')
    parser.add_argument('--output', type=str, default='models_output',
                       help='Output directory')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['MobileNet', 'EfficientNetB0', 'ResNet50'],
                       help='Models to train (space-separated)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Training epochs for frozen base')
    parser.add_argument('--fine_tune_epochs', type=int, default=10,
                       help='Fine-tuning epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--train_all', action='store_true',
                       help='Train all available models')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🐟 FISH CLASSIFICATION - ALL TRANSFER MODELS")
    print("="*70)
    
    # Initialize trainer
    trainer = TransferLearningTrainer(
        data_dir=args.data,
        output_dir=args.output,
        batch_size=args.batch
    )
    
    # Determine which models to train
    if args.train_all:
        models_to_train = list(trainer.available_models.keys())
    else:
        models_to_train = args.models
    
    # Start training
    start_time = time.time()
    
    results = trainer.train_all_models(
        models_list=models_to_train,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs
    )
    
    total_time = (time.time() - start_time) / 60
    
    print("\n" + "="*70)
    print(f"✅ ALL TRAINING COMPLETED IN {total_time:.2f} MINUTES")
    print("="*70)
    
    if results:
        print("\n📋 Quick Summary:")
        for r in results[:3]:  # Top 3 models
            print(f"  • {r['Model']}: {r['Accuracy']:.4f} accuracy")


if __name__ == "__main__":
    main()