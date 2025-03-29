# agents/anomaly_agent.py

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from sklearn.ensemble import IsolationForest
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.combination import aom, moa, average
from alibi_detect.od import OutlierProphet, OutlierVAE
from alibi_detect.utils.saving import save_detector, load_detector
import shap
import json
import os
import pickle
from datetime import datetime
from scipy.stats import zscore
from functools import partial
from agents.viz_agent import VizAgent
from agents.data_agent import DataAgent

class AnomalyAgent:
    """
    Advanced Anomaly Detection Agent with Ensemble Detection, Explainable AI,
    and Adaptive Thresholding capabilities.
    
    Features:
    - Multi-modal detection (tabular, time-series, NLP embeddings)
    - Hybrid unsupervised/semi-supervised approaches
    - Automatic concept drift detection
    - SHAP-based anomaly explanations
    - Dynamic threshold optimization
    - Collaborative filtering integration
    - Automated model versioning
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.detectors = {}
        self.explainers = {}
        self._init_directories()
        
        # Initialize SHAP
        self.explainer = shap.DeepExplainer
        
        # State management
        self.detection_history = []
        self.model_versions = {}
        self.feedback_buffer = []
        
        # Initialize sub-agents
        self.viz_agent = VizAgent()
        self.data_agent = DataAgent()

    def _init_directories(self):
        """Create required directory structure"""
        os.makedirs("models/anomaly", exist_ok=True)
        os.makedirs("data/anomalies", exist_ok=True)

    def run(self, **params) -> Dict[str, Any]:
        """Main execution interface for anomaly detection workflows"""
        operation = params.get("operation", "detect")
        
        try:
            if operation == "detect":
                return self.detect_anomalies(**params)
            elif operation == "explain":
                return self.explain_anomalies(**params)
            elif operation == "retrain":
                return self.retrain_model(**params)
            elif operation == "feedback":
                return self.process_feedback(**params)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        except Exception as e:
            self.logger.error(f"Anomaly operation failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def detect_anomalies(self, dataset_id: str, methods: List[str] = None,
                        mode: str = "ensemble") -> Dict[str, Any]:
        """
        Perform advanced anomaly detection using multiple detection strategies
        
        Args:
            dataset_id: ID of the processed dataset
            methods: List of detection methods to use
            mode: Detection strategy (ensemble, sequential, parallel)
            
        Returns:
            Dictionary with detection results and metadata
        """
        data = self.data_agent.get_dataset(dataset_id)
        metadata = self.data_agent.get_metadata(dataset_id)
        
        # Automatic method selection
        if not methods:
            methods = self._recommend_methods(data, metadata)
            
        # Multi-modal detection pipeline
        results = {}
        for method in methods:
            method_fn = getattr(self, f"_detect_{method}")
            results[method] = method_fn(data)
            
        # Combine results using advanced consensus
        combined = self._consensus_combination(results, mode=mode)
        
        # Generate comprehensive report
        report = self._generate_detection_report(data, combined, methods)
        
        # Visualize anomalies
        visualizations = self._visualize_anomalies(data, report)
        
        # Version and save results
        version_id = self._version_detection(report, methods)
        
        return {
            "success": True,
            "version_id": version_id,
            "report": report,
            "visualizations": visualizations,
            "metadata": {
                "dataset_id": dataset_id,
                "methods_used": methods,
                "mode": mode,
                "timestamp": datetime.now().isoformat()
            }
        }

    def _detect_deep_autoencoder(self, data: pd.DataFrame) -> Dict:
        """Deep learning-based anomaly detection using AutoEncoder"""
        detector = AutoEncoder(epochs=100, contamination=0.1)
        detector.fit(data)
        scores = detector.decision_scores_
        return self._format_scores(scores)

    def _detect_temporal_prophet(self, data: pd.DataFrame) -> Dict:
        """Time-series aware anomaly detection using Facebook Prophet"""
        detector = OutlierProphet(
            threshold=0.01,
            model=Prophet(),
            cap=1.5
        )
        detector.fit(data)
        preds = detector.predict(data)
        return self._format_scores(preds['data']['instance_score'])

    def _detect_adaptive_forest(self, data: pd.DataFrame) -> Dict:
        """Adaptive Isolation Forest with dynamic thresholding"""
        detector = IsolationForest(
            n_estimators=200,
            contamination='auto',
            behaviour='new'
        )
        detector.fit(data)
        scores = -detector.decision_function(data)
        return self._format_scores(scores)

    def _detect_univariate_zscore(self, data: pd.DataFrame) -> Dict:
        """Adaptive Z-Score with automatic threshold calibration"""
        scores = data.apply(zscore).abs()
        return self._format_scores(scores.max(axis=1))

    def _consensus_combination(self, results: Dict, mode: str = "ensemble") -> Dict:
        """Advanced result combination using multiple consensus strategies"""
        # Implement AOM (Average of Maximum) and MOA (Maximum of Average)
        scores = np.array([r['scores'] for r in results.values()])
        
        if mode == "ensemble":
            combined_scores = average(scores)
        elif mode == "aom":
            combined_scores = aom(scores, n_buckets=5)
        elif mode == "moa":
            combined_scores = moa(scores, n_buckets=5)
        else:
            raise ValueError(f"Unknown combination mode: {mode}")
            
        return {
            'scores': combined_scores,
            'threshold': self._dynamic_threshold(combined_scores),
            'methods': list(results.keys())
        }

    def _dynamic_threshold(self, scores: np.ndarray) -> float:
        """Calculate dynamic threshold using modified IQR with outlier resistance"""
        q75, q25 = np.percentile(scores, [75, 25])
        iqr = q75 - q25
        return q75 + (1.5 * iqr) * (1 + np.log1p(len(scores)/1000))

    def explain_anomalies(self, dataset_id: str, version_id: str) -> Dict:
        """Generate SHAP-based explanations for detected anomalies"""
        data = self.data_agent.get_dataset(dataset_id)
        detection = self._load_detection_version(version_id)
        
        # Train explainer on normal data
        normal_data = data[~detection['anomaly_flags']]
        background = normal_data.sample(min(1000, len(normal_data)))
        
        # Explain anomalies
        explainer = self.explainer(
            partial(self._detect_deep_autoencoder, return_scores=True),
            background.values
        )
        shap_values = explainer.shap_values(data.values)
        
        return {
            "global_importance": self._aggregate_shap(shap_values),
            "local_explanations": [
                {"index": i, "features": self._format_shap(features)}
                for i, features in enumerate(shap_values)
                if detection['anomaly_flags'][i]
            ]
        }

    def process_feedback(self, feedback: Dict) -> Dict:
        """Incorporate human feedback into anomaly detection models"""
        self.feedback_buffer.append(feedback)
        
        if len(self.feedback_buffer) >= self.config.get('retrain_batch', 50):
            return self.retrain_model()
            
        return {"status": "feedback_stored", "buffer_size": len(self.feedback_buffer)}

    def retrain_model(self) -> Dict:
        """Active learning retraining with human feedback"""
        if not self.feedback_buffer:
            return {"success": False, "error": "No feedback available"}
            
        # Create augmented training data
        feedback_data = self._process_feedback_data()
        
        # Retrain detectors with new data
        for name, detector in self.detectors.items():
            detector.fit(feedback_data['X'], feedback_data['y'])
            
        # Clear feedback buffer
        self.feedback_buffer = []
        
        return {"success": True, "retrained_models": list(self.detectors.keys())}

    def _version_detection(self, report: Dict, methods: List[str]) -> str:
        """Version control for detection results"""
        version_id = f"AD_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        path = f"data/anomalies/{version_id}"
        
        os.makedirs(path)
        with open(f"{path}/report.json", "w") as f:
            json.dump(report, f)
            
        self.detection_history.append(version_id)
        return version_id

    def _visualize_anomalies(self, data: pd.DataFrame, report: Dict) -> Dict:
        """Generate interactive visualizations of anomalies"""
        viz_data = data.copy()
        viz_data['anomaly_score'] = report['scores']
        viz_data['is_anomaly'] = report['anomaly_flags']
        
        return {
            "scatter_matrix": self.viz_agent.create_visualization(
                viz_data, "scatter", color="is_anomaly", title="Anomaly Distribution"
            ),
            "score_distribution": self.viz_agent.create_visualization(
                viz_data, "histogram", x="anomaly_score", title="Anomaly Scores"
            ),
            "temporal_trend": self.viz_agent.create_visualization(
                viz_data, "line", x="timestamp", y="anomaly_score", 
                color="is_anomaly", title="Temporal Anomaly Trend"
            ) if 'timestamp' in viz_data else None
        }

    def _recommend_methods(self, data: pd.DataFrame, metadata: Dict) -> List[str]:
        """Automatically recommend detection methods based on data characteristics"""
        methods = []
        
        if metadata.get('temporal', False):
            methods.append('temporal_prophet')
            
        if len(data.columns) > 3:
            methods.extend(['deep_autoencoder', 'adaptive_forest'])
        else:
            methods.extend(['univariate_zscore', 'adaptive_forest'])
            
        return methods

    # Helper methods and additional detection algorithms omitted for brevity

    def _format_scores(self, scores: np.ndarray) -> Dict:
        """Standardize score formatting"""
        return {
            'scores': scores.tolist(),
            'threshold': self._dynamic_threshold(scores),
            'anomaly_flags': scores > self._dynamic_threshold(scores)
        }

    def _aggregate_shap(self, shap_values: np.ndarray) -> List:
        """Aggregate SHAP values across all anomalies"""
        return pd.DataFrame(shap_values, columns=self.data.columns).abs().mean().to_dict()

    def _format_shap(self, shap_values: np.ndarray) -> List[Dict]:
        """Format SHAP values for individual explanations"""
        return [{
            "feature": col,
            "value": val,
            "shap_impact": float(shap_values[i])
        } for i, (col, val) in enumerate(zip(self.data.columns, row))] 
