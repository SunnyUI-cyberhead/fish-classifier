"""
Fish Image Classification - Model Evaluation and Comparison Script
This script provides comprehensive evaluation metrics and comparison tools
for all trained models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ModelEvaluator:
    def __init__(self, models_dir='models', data_dir='fish_dataset', img_size=(224, 224)):
        """
        Initialize the Model Evaluator
        
        Args:
            models_dir: Directory containing trained models
            data_dir: Directory containing the dataset
            img_size: Image size used for training
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.models = {}
        self.results = {}
        self.class_names = None
        self.num_classes = None
        
    def load_all_models(self):
        """Load all trained models from the models directory"""
        print("Loading all trained models...")
        print("="*50)
        
        model_files = list(self.models_dir.glob("*_best.h5"))
        
        if not model_files:
            model_files = list(self.models_dir.glob("*_final.h5"))
        
        for model_path in model_files:
            model_name = model_path.stem.replace('_best', '').replace('_final', '')
            try:
                print(f"Loading {model_name}...")
                self.models[model_name] = load_model(model_path)
                print(f"✓ {model_name} loaded successfully")
            except Exception as e:
                print(f"✗ Error loading {model_name}: {str(e)}")
        
        print(f"\nTotal models loaded: {len(self.models)}")
        return self.models
    
    def prepare_test_data(self, test_split=0.2):
        """Prepare test data generator"""
        print("\nPreparing test data...")
        
        # Load class indices
        class_indices_path = self.models_dir / 'class_indices.json'
        if class_indices_path.exists():
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
            self.class_names = list(class_indices.keys())
            self.num_classes = len(self.class_names)
        
        # Create test data generator
        test_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=test_split
        )
        
        self.test_generator = test_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        print(f"Test samples: {self.test_generator.samples}")
        print(f"Number of classes: {self.num_classes}")
        
        return self.test_generator
    
    def evaluate_single_model(self, model_name, model):
        """
        Comprehensive evaluation of a single model
        """
        print(f"\nEvaluating {model_name}...")
        print("-"*30)
        
        # Get predictions
        predictions = model.predict(self.test_generator, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes
        
        # Calculate metrics
        test_loss, test_acc, test_precision, test_recall = model.evaluate(
            self.test_generator, 
            verbose=0
        )
        
        # Classification report
        report = classification_report(
            y_true, 
            y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        per_class_metrics = []
        for class_name in self.class_names:
            if class_name in report:
                per_class_metrics.append({
                    'class': class_name,
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1-score': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                })
        
        # Store results
        self.results[model_name] = {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'weighted_f1': report['weighted avg']['f1-score'],
            'macro_f1': report['macro avg']['f1-score'],
            'confusion_matrix': cm,
            'classification_report': report,
            'per_class_metrics': per_class_metrics,
            'predictions': predictions
        }
        
        print(f"Accuracy: {test_acc:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall: {test_recall:.4f}")
        print(f"F1-Score (weighted): {report['weighted avg']['f1-score']:.4f}")
        
        return self.results[model_name]
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        if n_models == 0:
            print("No models to plot. Evaluate models first.")
            return
        
        fig, axes = plt.subplots(
            2, 3, 
            figsize=(18, 12)
        )
        axes = axes.flatten()
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            if idx >= 6:  # Only plot first 6 models
                break
                
            cm = results['confusion_matrix']
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=axes[idx],
                cbar_kws={'label': 'Normalized Count'}
            )
            axes[idx].set_title(f'{model_name}\nAccuracy: {results["test_accuracy"]:.3f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')
            
            # Rotate labels for better readability
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right')
            axes[idx].set_yticklabels(axes[idx].get_yticklabels(), rotation=0)
        
        # Hide unused subplots
        for idx in range(n_models, 6):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.models_dir / 'all_confusion_matrices.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_comparison(self):
        """Create comprehensive metrics comparison plots"""
        if not self.results:
            print("No results to plot. Evaluate models first.")
            return
        
        # Prepare data for plotting
        metrics_data = []
        for model_name, results in self.results.items():
            metrics_data.append({
                'Model': model_name,
                'Accuracy': results['test_accuracy'],
                'Precision': results['test_precision'],
                'Recall': results['test_recall'],
                'F1-Score (Weighted)': results['weighted_f1'],
                'F1-Score (Macro)': results['macro_f1'],
                'Loss': results['test_loss']
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Plot 1: Accuracy comparison
        axes[0, 0].bar(df_metrics['Model'], df_metrics['Accuracy'], color='skyblue')
        axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(df_metrics['Accuracy']):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Plot 2: Precision comparison
        axes[0, 1].bar(df_metrics['Model'], df_metrics['Precision'], color='lightcoral')
        axes[0, 1].set_title('Model Precision Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for i, v in enumerate(df_metrics['Precision']):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Plot 3: Recall comparison
        axes[0, 2].bar(df_metrics['Model'], df_metrics['Recall'], color='lightgreen')
        axes[0, 2].set_title('Model Recall Comparison', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Model')
        axes[0, 2].set_ylabel('Recall')
        axes[0, 2].set_ylim([0, 1])
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        for i, v in enumerate(df_metrics['Recall']):
            axes[0, 2].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Plot 4: F1-Score comparison
        x = np.arange(len(df_metrics['Model']))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, df_metrics['F1-Score (Weighted)'], width, label='Weighted', color='gold')
        axes[1, 0].bar(x + width/2, df_metrics['F1-Score (Macro)'], width, label='Macro', color='orange')
        axes[1, 0].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(df_metrics['Model'], rotation=45)
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 1])
        
        # Plot 5: Loss comparison
        axes[1, 1].bar(df_metrics['Model'], df_metrics['Loss'], color='salmon')
        axes[1, 1].set_title('Model Loss Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        for i, v in enumerate(df_metrics['Loss']):
            axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Plot 6: Radar chart for best model
        best_model = df_metrics.loc[df_metrics['Accuracy'].idxmax(), 'Model']
        best_metrics = df_metrics[df_metrics['Model'] == best_model].iloc[0]
        
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score\n(Weighted)', 'F1-Score\n(Macro)']
        values = [
            best_metrics['Accuracy'],
            best_metrics['Precision'],
            best_metrics['Recall'],
            best_metrics['F1-Score (Weighted)'],
            best_metrics['F1-Score (Macro)']
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values_plot = values + values[:1]  # Complete the circle
        angles += angles[:1]
        
        axes[1, 2].remove()  # Remove the cartesian axis
        ax_polar = fig.add_subplot(2, 3, 6, projection='polar')
        ax_polar.plot(angles, values_plot, 'o-', linewidth=2, color='darkblue')
        ax_polar.fill(angles, values_plot, alpha=0.25, color='darkblue')
        ax_polar.set_xticks(angles[:-1])
        ax_polar.set_xticklabels(categories)
        ax_polar.set_ylim(0, 1)
        ax_polar.set_title(f'Best Model: {best_model}', fontsize=14, fontweight='bold', pad=20)
        ax_polar.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.models_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return df_metrics
    
    def plot_per_class_performance(self):
        """Plot per-class performance for each model"""
        if not self.results:
            print("No results to plot. Evaluate models first.")
            return
        
        for model_name, results in self.results.items():
            per_class_df = pd.DataFrame(results['per_class_metrics'])
            
            if per_class_df.empty:
                continue
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'{model_name} - Per-Class Performance', fontsize=16, fontweight='bold')
            
            # Precision per class
            axes[0].barh(per_class_df['class'], per_class_df['precision'], color='lightcoral')
            axes[0].set_xlabel('Precision')
            axes[0].set_title('Precision by Class')
            axes[0].set_xlim([0, 1])
            
            # Recall per class
            axes[1].barh(per_class_df['class'], per_class_df['recall'], color='lightgreen')
            axes[1].set_xlabel('Recall')
            axes[1].set_title('Recall by Class')
            axes[1].set_xlim([0, 1])
            
            # F1-Score per class
            axes[2].barh(per_class_df['class'], per_class_df['f1-score'], color='skyblue')
            axes[2].set_xlabel('F1-Score')
            axes[2].set_title('F1-Score by Class')
            axes[2].set_xlim([0, 1])
            
            plt.tight_layout()
            plt.savefig(self.models_dir / f'{model_name}_per_class_performance.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    def generate_comparison_report(self):
        """Generate a comprehensive comparison report"""
        if not self.results:
            print("No results to generate report. Evaluate models first.")
            return
        
        # Create summary DataFrame
        summary_data = []
        for model_name, results in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Test Accuracy': f"{results['test_accuracy']:.4f}",
                'Test Precision': f"{results['test_precision']:.4f}",
                'Test Recall': f"{results['test_recall']:.4f}",
                'F1-Score (Weighted)': f"{results['weighted_f1']:.4f}",
                'F1-Score (Macro)': f"{results['macro_f1']:.4f}",
                'Test Loss': f"{results['test_loss']:.4f}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('Test Accuracy', ascending=False)
        
        # Generate report
        report_path = self.models_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FISH CLASSIFICATION MODEL EVALUATION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write("SUMMARY OF MODEL PERFORMANCE\n")
            f.write("-"*40 + "\n")
            f.write(df_summary.to_string(index=False))
            f.write("\n\n")
            
            # Best model
            best_model = df_summary.iloc[0]['Model']
            f.write(f"BEST PERFORMING MODEL: {best_model}\n")
            f.write("-"*40 + "\n")
            
            best_results = self.results[best_model]
            f.write(f"Accuracy: {best_results['test_accuracy']:.4f}\n")
            f.write(f"Precision: {best_results['test_precision']:.4f}\n")
            f.write(f"Recall: {best_results['test_recall']:.4f}\n")
            f.write(f"F1-Score (Weighted): {best_results['weighted_f1']:.4f}\n")
            f.write(f"F1-Score (Macro): {best_results['macro_f1']:.4f}\n\n")
            
            # Detailed classification reports
            f.write("\nDETAILED CLASSIFICATION REPORTS\n")
            f.write("="*80 + "\n")
            
            for model_name in df_summary['Model']:
                f.write(f"\n{model_name}\n")
                f.write("-"*40 + "\n")
                
                # Convert classification report to string
                report_dict = self.results[model_name]['classification_report']
                report_str = classification_report(
                    self.test_generator.classes,
                    np.argmax(self.results[model_name]['predictions'], axis=1),
                    target_names=self.class_names,
                    digits=4
                )
                f.write(report_str)
                f.write("\n")
        
        print(f"\n✓ Evaluation report saved to: {report_path}")
        
        # Save results as pickle for later use
        results_path = self.models_dir / 'evaluation_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"✓ Evaluation results saved to: {results_path}")
        
        # Save summary as CSV
        csv_path = self.models_dir / 'model_comparison_summary.csv'
        df_summary.to_csv(csv_path, index=False)
        print(f"✓ Summary CSV saved to: {csv_path}")
        
        return df_summary
    
    def run_complete_evaluation(self):
        """Run complete evaluation pipeline"""
        print("\n" + "="*60)
        print("STARTING COMPLETE MODEL EVALUATION")
        print("="*60)
        
        # Load models
        self.load_all_models()
        
        if not self.models:
            print("No models found for evaluation!")
            return
        
        # Prepare test data
        self.prepare_test_data()
        
        # Evaluate each model
        for model_name, model in self.models.items():
            self.evaluate_single_model(model_name, model)
        
        # Generate visualizations
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        self.plot_confusion_matrices()
        metrics_df = self.plot_metrics_comparison()
        self.plot_per_class_performance()
        
        # Generate report
        print("\n" + "="*60)
        print("GENERATING EVALUATION REPORT")
        print("="*60)
        
        summary_df = self.generate_comparison_report()
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        
        print("\nFINAL RANKINGS:")
        print(summary_df[['Model', 'Test Accuracy']].to_string(index=False))
        
        return summary_df, self.results

def main():
    """Main execution function"""
    # Configuration
    MODELS_DIR = 'models'
    DATA_DIR = r"D:\Machine Learning\Fish Classification\Dataset\images"  # Update this path to your dataset
    IMG_SIZE = (224, 224)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        models_dir=MODELS_DIR,
        data_dir=DATA_DIR,
        img_size=IMG_SIZE
    )
    
    # Run complete evaluation
    summary, results = evaluator.run_complete_evaluation()
    
    print("\n✅ All evaluations completed successfully!")
    print(f"Check the '{MODELS_DIR}' directory for all outputs and reports.")

if __name__ == "__main__":
    main()