"""
Fish Image Classification - Ultimate Fix for Overfitting
Addresses severe overfitting with aggressive regularization and techniques
"""

import os
import argparse
import json
import time
from pathlib import Path
from collections import Counter
import hashlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# CRITICAL: Limit GPU memory growth to prevent issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def md5_hash(path):
    """Calculate MD5 hash of a file"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_image_hashes(root):
    """Collect MD5 hashes of all images in directory"""
    root = Path(root)
    by_hash = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            try:
                by_hash.setdefault(md5_hash(str(p)), []).append(str(p))
            except Exception:
                pass
    return by_hash


class LabelSmoothingCrossEntropy(tf.keras.losses.Loss):
    """Label smoothing to prevent overconfidence"""
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        num_classes = tf.shape(y_pred)[-1]
        y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / tf.cast(num_classes, tf.float32))
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)


class FishClassifierTrainer:
    def __init__(self, data_dir, output_dir="models_output", img_size=(224, 224), batch_size=8):
        self.data_root = Path(data_dir)
        self.train_dir = self.data_root / "train"
        self.val_dir = self.data_root / "val"
        self.test_dir = self.data_root / "test"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        self.img_size = tuple(img_size)
        self.batch_size = batch_size  # Smaller batch size
        
        self.num_classes = None
        self.class_names = None
        self.class_weights = None
        
        self.history_dict = {}
        self.models_dict = {}
        self.results_dict = {}
        
        print(f"📁 Dataset root: {self.data_root}")
        print(f"💾 Output dir: {self.output_dir}")
        print(f"📦 Batch size: {self.batch_size} (small to prevent overfitting)")
        
        self.validate_dataset()
        self.check_duplicate_leakage()

    def validate_dataset(self):
        """Validate dataset structure and provide warnings"""
        if not self.train_dir.exists() or not self.val_dir.exists():
            raise ValueError(
                f"Expected train/ and val/ directories in {self.data_root}\n"
                f"Found: train={self.train_dir.exists()}, val={self.val_dir.exists()}"
            )
        
        def count_images(folder):
            cnt = 0
            per_class = {}
            for cls in sorted([d for d in Path(folder).iterdir() if d.is_dir()]):
                imgs = []
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                    imgs.extend(cls.glob(ext))
                per_class[cls.name] = len(imgs)
                cnt += len(imgs)
            return cnt, per_class
        
        tr_total, tr_counts = count_images(self.train_dir)
        va_total, va_counts = count_images(self.val_dir)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  • Train: {tr_total} images across {len(tr_counts)} classes")
        print(f"  • Val: {va_total} images across {len(va_counts)} classes")
        
        # Critical warnings
        if tr_total < 500:
            print(f"\n⚠️  CRITICAL WARNING: Very small training set ({tr_total} images)!")
            print("   This dataset is likely TOO SMALL for deep learning.")
            print("   Expect severe overfitting. Consider:")
            print("   - Collecting more data (aim for 1000+ images)")
            print("   - Using traditional ML (Random Forest, SVM)")
            print("   - Using extreme regularization (implemented below)")
        
        min_images = min(tr_counts.values()) if tr_counts else 0
        if min_images < 50:
            print(f"\n⚠️  WARNING: Some classes have very few images (min: {min_images})")
            print("   Model will likely memorize these classes.")

    def check_duplicate_leakage(self):
        """Check for data leakage between splits"""
        print("\n🔍 Checking for data leakage...")
        tr_hashes = collect_image_hashes(self.train_dir)
        va_hashes = collect_image_hashes(self.val_dir)
        overlap = set(tr_hashes.keys()) & set(va_hashes.keys())
        
        if overlap:
            print(f"❌ CRITICAL: Found {len(overlap)} duplicate files between train and val!")
            for i, h in enumerate(list(overlap)[:5], 1):
                print(f"   Example {i}: {tr_hashes[h][0]} <-> {va_hashes[h][0]}")
            raise ValueError("Data leakage detected! Remove duplicates before training.")
        print("✅ No data leakage detected")

    def prepare_data(self):
        """Prepare data with EXTREME augmentation to combat overfitting"""
        print("\n📊 Preparing data with aggressive augmentation...")
        
        # EXTREME augmentation for small datasets
        train_datagen = ImageDataGenerator(
            rescale=1.0/255,
            rotation_range=40,        # High rotation
            width_shift_range=0.3,    # High shift
            height_shift_range=0.3,   # High shift
            zoom_range=0.4,           # High zoom
            shear_range=0.3,          # High shear
            horizontal_flip=True,
            vertical_flip=True,       # Also vertical
            brightness_range=[0.5, 1.5],  # Strong brightness changes
            channel_shift_range=50,   # Color changes
            fill_mode="reflect"
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator(rescale=1.0/255)
        
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )
        
        self.val_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False
        )
        
        self.num_classes = len(self.train_generator.class_indices)
        self.class_names = list(self.train_generator.class_indices.keys())
        
        # Calculate class weights
        counter = Counter(self.train_generator.classes)
        max_val = float(max(counter.values()))
        self.class_weights = {cls: max_val/n for cls, n in counter.items()}
        
        # Save class indices
        with open(self.output_dir / "class_indices.json", "w") as f:
            json.dump(self.train_generator.class_indices, f, indent=2)
        
        print(f"✅ Data prepared:")
        print(f"   • Classes: {self.num_classes}")
        print(f"   • Train samples: {self.train_generator.samples}")
        print(f"   • Val samples: {self.val_generator.samples}")
        print(f"   • Using class weights: Yes")
        print(f"   • Using label smoothing: Yes (0.1)")

    def build_ultra_simple_cnn(self):
        """Build ULTRA SIMPLE CNN for tiny datasets"""
        print("\n🔨 Building ULTRA SIMPLE CNN (for small datasets)...")
        
        model = models.Sequential([
            # Input with strong noise
            layers.Input(shape=(*self.img_size, 3)),
            layers.GaussianNoise(0.2),  # Strong noise
            
            # Very simple architecture
            layers.Conv2D(8, 5, activation="relu", padding="same",
                         kernel_regularizer=regularizers.l2(0.05)),  # Strong L2
            layers.MaxPooling2D(4),  # Aggressive pooling
            layers.Dropout(0.5),     # High dropout
            
            layers.Conv2D(16, 3, activation="relu", padding="same",
                         kernel_regularizer=regularizers.l2(0.05)),
            layers.MaxPooling2D(2),
            layers.Dropout(0.6),     # Very high dropout
            
            # Simple classifier
            layers.Flatten(),
            layers.Dense(32, activation="relu",
                        kernel_regularizer=regularizers.l2(0.05)),
            layers.Dropout(0.7),     # Extreme dropout
            layers.Dense(self.num_classes, activation="softmax")
        ])
        
        print(f"   ✓ Ultra simple model with only {model.count_params():,} parameters")
        return model

    def build_regularized_transfer_model(self, base_name="MobileNet"):
        """Build heavily regularized transfer model"""
        print(f"\n🔨 Building heavily regularized {base_name}...")
        
        from tensorflow.keras.applications import MobileNet, EfficientNetB0
        
        if base_name == "MobileNet":
            base_model = MobileNet(weights="imagenet", include_top=False, 
                                  input_shape=(*self.img_size, 3))
        elif base_name == "EfficientNetB0":
            base_model = EfficientNetB0(weights="imagenet", include_top=False,
                                       input_shape=(*self.img_size, 3))
        else:
            raise ValueError(f"Unsupported model: {base_name}")
        
        # FREEZE EVERYTHING - crucial for small datasets
        base_model.trainable = False
        
        # Build model with extreme regularization
        inputs = layers.Input(shape=(*self.img_size, 3))
        
        # Add multiple regularization layers
        x = layers.GaussianNoise(0.3)(inputs)  # Strong noise
        x = layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(x)
        x = layers.experimental.preprocessing.RandomRotation(0.2)(x)
        x = layers.experimental.preprocessing.RandomZoom(0.2)(x)
        
        # Base model
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Ultra simple head with extreme dropout
        x = layers.Dropout(0.7)(x)  # Extreme dropout
        x = layers.Dense(32, activation="relu",
                        kernel_regularizer=regularizers.l2(0.1))(x)  # Strong L2
        x = layers.Dropout(0.8)(x)  # Even more dropout
        
        outputs = layers.Dense(self.num_classes, activation="softmax",
                              kernel_regularizer=regularizers.l2(0.1))(x)
        
        model = models.Model(inputs, outputs)
        
        print(f"   ✓ Model with {model.count_params():,} parameters")
        print(f"   ✓ Trainable: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        
        return model

    def compile_model(self, model, lr=0.00001):  # VERY low learning rate
        """Compile with label smoothing and low learning rate"""
        model.compile(
            optimizer=Adam(learning_rate=lr, clipnorm=1.0),  # Gradient clipping
            loss=LabelSmoothingCrossEntropy(label_smoothing=0.1),  # Label smoothing
            metrics=["accuracy"]
        )
        return model

    def get_callbacks(self, name):
        """Callbacks with aggressive early stopping"""
        return [
            callbacks.ModelCheckpoint(
                filepath=str(self.output_dir / f"{name}_best.h5"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,  # Very aggressive early stopping
                restore_best_weights=True,
                verbose=1,
                min_delta=0.01  # Large delta required
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,  # Aggressive reduction
                patience=2,
                min_lr=1e-8,
                verbose=1
            ),
            # Custom callback to stop if accuracy is suspiciously high
            callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: 
                    print("\n⚠️  WARNING: Perfect training accuracy - severe overfitting!")
                    if logs.get('accuracy', 0) > 0.99 else None
            )
        ]

    def train_with_monitoring(self, model, name, epochs=30):
        """Train with overfitting monitoring"""
        print(f"\n{'='*60}")
        print(f"🚀 Training {name} with anti-overfitting measures")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Train with all regularization
        history = model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            class_weight=self.class_weights,
            callbacks=self.get_callbacks(name),
            verbose=1
        )
        
        # Save model and history
        model.save(self.output_dir / f"{name}_final.h5")
        self.history_dict[name] = history.history
        self.models_dict[name] = model
        
        # Check for overfitting
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        gap = abs(final_train_acc - final_val_acc)
        
        print(f"\n📊 Training Summary:")
        print(f"   • Time: {(time.time() - start_time)/60:.2f} minutes")
        print(f"   • Final train accuracy: {final_train_acc:.4f}")
        print(f"   • Final val accuracy: {final_val_acc:.4f}")
        print(f"   • Overfitting gap: {gap:.4f}")
        
        if final_train_acc > 0.95:
            print("\n⚠️  SEVERE OVERFITTING DETECTED!")
            print("   Your dataset is too small/simple for deep learning.")
            print("   Consider using traditional ML methods instead.")
        elif gap > 0.2:
            print("\n⚠️  Significant overfitting detected.")
        
        return history

    def evaluate_model(self, model, name):
        """Evaluate with detailed metrics"""
        print(f"\n📊 Evaluating {name}...")
        
        val_loss, val_acc = model.evaluate(self.val_generator, verbose=0)
        
        # Get predictions
        preds = model.predict(self.val_generator, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = self.val_generator.classes
        
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
        self.results_dict[name] = {
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "f1_weighted": float(report["weighted avg"]["f1-score"]),
            "confusion_matrix": cm.tolist()
        }
        
        print(f"   • Val Accuracy: {val_acc:.4f}")
        print(f"   • Val Loss: {val_loss:.4f}")
        print(f"   • F1 Score: {report['weighted avg']['f1-score']:.4f}")
        
        if val_acc > 0.98:
            print("\n   ⚠️  Suspiciously high accuracy - check for data issues!")
        
        return self.results_dict[name]

    def plot_training_curves(self, name):
        """Plot training curves with overfitting indicators"""
        history = self.history_dict[name]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        epochs_range = range(1, len(history['accuracy']) + 1)
        axes[0].plot(epochs_range, history['accuracy'], 'b-', label='Training', linewidth=2)
        axes[0].plot(epochs_range, history['val_accuracy'], 'r-', label='Validation', linewidth=2)
        axes[0].fill_between(epochs_range, history['accuracy'], history['val_accuracy'],
                            alpha=0.3, color='yellow', label='Overfitting Gap')
        axes[0].set_title(f'{name} - Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.05])
        
        # Loss plot
        axes[1].plot(epochs_range, history['loss'], 'b-', label='Training', linewidth=2)
        axes[1].plot(epochs_range, history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[1].set_title(f'{name} - Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{name}_curves.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        # Print gap analysis
        final_gap = abs(history['accuracy'][-1] - history['val_accuracy'][-1])
        print(f"\n📈 Overfitting Analysis:")
        print(f"   • Final accuracy gap: {final_gap:.4f}")
        if final_gap > 0.15:
            print(f"   • ⚠️  Model is overfitting significantly!")

    def quick_train(self, model_type="ultra_simple", epochs=20):
        """Quick training with full pipeline"""
        # Prepare data
        self.prepare_data()
        
        # Build model based on type
        if model_type == "ultra_simple":
            model = self.build_ultra_simple_cnn()
            name = "UltraSimpleCNN"
        elif model_type in ["MobileNet", "EfficientNetB0"]:
            model = self.build_regularized_transfer_model(model_type)
            name = f"Regularized_{model_type}"
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Compile with very low learning rate
        model = self.compile_model(model, lr=0.00001)
        
        # Train with monitoring
        history = self.train_with_monitoring(model, name, epochs)
        
        # Evaluate
        self.evaluate_model(model, name)
        
        # Plot curves
        self.plot_training_curves(name)
        
        # Save results
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(self.results_dict, f, indent=2)
        
        return model


def main():
    parser = argparse.ArgumentParser(description="Fish Classification - Ultimate Overfitting Fix")
    parser.add_argument("--data", type=str, 
                       default=r"D:\Machine Learning\Fish Classification\Dataset\images\data",
                       help="Path to dataset with train/val/test folders")
    parser.add_argument("--output", type=str, default="models_output",
                       help="Output directory")
    parser.add_argument("--model", type=str, default="ultra_simple",
                       choices=["ultra_simple", "MobileNet", "EfficientNetB0"],
                       help="Model type to use")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Training epochs")
    parser.add_argument("--batch", type=int, default=8,
                       help="Batch size (keep small for regularization)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("🐟 FISH CLASSIFICATION - OVERFITTING FIX")
    print("="*60)
    print("\n🛡️ Anti-overfitting measures enabled:")
    print("   • Label smoothing (0.1)")
    print("   • Extreme dropout (up to 0.8)")
    print("   • Strong L2 regularization (0.05-0.1)")
    print("   • Aggressive data augmentation")
    print("   • Very low learning rate (1e-5)")
    print("   • Small batch size (8)")
    print("   • Gradient clipping")
    print("   • Early stopping (patience=3)")
    
    # Initialize trainer
    trainer = FishClassifierTrainer(
        data_dir=args.data,
        output_dir=args.output,
        batch_size=args.batch
    )
    
    # Train model
    start = time.time()
    model = trainer.quick_train(model_type=args.model, epochs=args.epochs)
    elapsed = (time.time() - start) / 60.0
    
    print("\n" + "="*60)
    print(f"✅ Training complete in {elapsed:.2f} minutes")
    print("="*60)
    
    # Final recommendations
    if trainer.results_dict:
        for name, results in trainer.results_dict.items():
            if results['val_accuracy'] > 0.98:
                print("\n🔴 CRITICAL FINDING:")
                print("   Your model STILL achieved near-perfect accuracy.")
                print("   This means your dataset is:")
                print("   1. Too small (< 500 images)")
                print("   2. Too simple (classes are trivially different)")
                print("   3. Has data quality issues")
                print("\n   STRONG RECOMMENDATION:")
                print("   • Use Random Forest or SVM instead of deep learning")
                print("   • Or collect MUCH more diverse data (5000+ images)")
                break


if __name__ == "__main__":
    main()