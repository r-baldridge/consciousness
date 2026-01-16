# Recurrent Processing Performance Metrics

## Performance Measurement Framework

### Core Performance Metrics
```python
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import numpy as np
import asyncio
from collections import deque

class MetricType(Enum):
    PROCESSING_LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    RESOURCE_UTILIZATION = "resource"
    CONSCIOUSNESS_DETECTION = "consciousness"
    INTEGRATION_PERFORMANCE = "integration"
    TEMPORAL_CONSISTENCY = "temporal"

class PerformanceLevel(Enum):
    OPTIMAL = "optimal"      # > 95th percentile
    EXCELLENT = "excellent"  # 90-95th percentile
    GOOD = "good"           # 75-90th percentile
    ACCEPTABLE = "acceptable"# 50-75th percentile
    POOR = "poor"           # < 50th percentile

@dataclass
class PerformanceMetric:
    metric_type: MetricType
    value: float
    unit: str
    timestamp: float
    context: Dict = field(default_factory=dict)
    percentile_rank: Optional[float] = None
    performance_level: Optional[PerformanceLevel] = None

@dataclass
class PerformanceBenchmark:
    metric_type: MetricType
    target_value: float
    threshold_excellent: float
    threshold_good: float
    threshold_acceptable: float
    unit: str
    description: str

class RecurrentProcessingPerformanceMonitor:
    """
    Comprehensive performance monitoring for recurrent processing system.
    """

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.benchmarks = self._initialize_benchmarks()
        self.metric_history = {metric_type: deque(maxlen=1000) for metric_type in MetricType}
        self.real_time_metrics = {}
        self.performance_stats = {}

    def _default_config(self) -> Dict:
        return {
            'monitoring_enabled': True,
            'real_time_collection': True,
            'history_retention_samples': 1000,
            'percentile_calculation_interval': 100,
            'alert_threshold_degradation': 0.2,
            'benchmark_update_frequency': 1000
        }

    def _initialize_benchmarks(self) -> Dict[MetricType, PerformanceBenchmark]:
        """Initialize performance benchmarks for recurrent processing."""
        return {
            MetricType.PROCESSING_LATENCY: PerformanceBenchmark(
                metric_type=MetricType.PROCESSING_LATENCY,
                target_value=50.0,      # Target: 50ms total processing
                threshold_excellent=75.0,
                threshold_good=100.0,
                threshold_acceptable=150.0,
                unit="milliseconds",
                description="End-to-end processing latency for single cycle"
            ),
            MetricType.THROUGHPUT: PerformanceBenchmark(
                metric_type=MetricType.THROUGHPUT,
                target_value=20.0,      # Target: 20 operations per second
                threshold_excellent=15.0,
                threshold_good=10.0,
                threshold_acceptable=5.0,
                unit="operations/second",
                description="Number of processing cycles per second"
            ),
            MetricType.ACCURACY: PerformanceBenchmark(
                metric_type=MetricType.ACCURACY,
                target_value=0.95,      # Target: 95% accuracy
                threshold_excellent=0.90,
                threshold_good=0.85,
                threshold_acceptable=0.80,
                unit="ratio",
                description="Overall processing accuracy across all stages"
            ),
            MetricType.CONSCIOUSNESS_DETECTION: PerformanceBenchmark(
                metric_type=MetricType.CONSCIOUSNESS_DETECTION,
                target_value=0.92,      # Target: 92% consciousness detection accuracy
                threshold_excellent=0.88,
                threshold_good=0.82,
                threshold_acceptable=0.75,
                unit="ratio",
                description="Accuracy of consciousness threshold detection"
            ),
            MetricType.RESOURCE_UTILIZATION: PerformanceBenchmark(
                metric_type=MetricType.RESOURCE_UTILIZATION,
                target_value=0.70,      # Target: 70% resource utilization
                threshold_excellent=0.80,
                threshold_good=0.85,
                threshold_acceptable=0.90,
                unit="ratio",
                description="CPU and memory utilization efficiency"
            ),
            MetricType.INTEGRATION_PERFORMANCE: PerformanceBenchmark(
                metric_type=MetricType.INTEGRATION_PERFORMANCE,
                target_value=25.0,      # Target: 25ms integration latency
                threshold_excellent=40.0,
                threshold_good=60.0,
                threshold_acceptable=100.0,
                unit="milliseconds",
                description="Integration communication latency"
            ),
            MetricType.TEMPORAL_CONSISTENCY: PerformanceBenchmark(
                metric_type=MetricType.TEMPORAL_CONSISTENCY,
                target_value=0.95,      # Target: 95% temporal consistency
                threshold_excellent=0.90,
                threshold_good=0.85,
                threshold_acceptable=0.80,
                unit="ratio",
                description="Consistency of timing across processing cycles"
            )
        }

    async def measure_processing_cycle_performance(self,
                                                 processing_input: Dict,
                                                 processing_function) -> Dict[MetricType, PerformanceMetric]:
        """
        Measure comprehensive performance metrics for a single processing cycle.

        Args:
            processing_input: Input data for processing
            processing_function: Function to measure

        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        start_time = time.time()

        # Resource monitoring
        initial_resources = await self._capture_resource_usage()

        try:
            # Execute processing with timing
            result = await self._execute_with_timing(processing_function, processing_input)

            # Calculate processing latency
            end_time = time.time()
            processing_latency = (end_time - start_time) * 1000  # Convert to milliseconds

            metrics[MetricType.PROCESSING_LATENCY] = PerformanceMetric(
                metric_type=MetricType.PROCESSING_LATENCY,
                value=processing_latency,
                unit="milliseconds",
                timestamp=end_time,
                context={'input_size': len(str(processing_input))}
            )

            # Calculate accuracy if ground truth available
            if 'ground_truth' in processing_input:
                accuracy = self._calculate_accuracy(result, processing_input['ground_truth'])
                metrics[MetricType.ACCURACY] = PerformanceMetric(
                    metric_type=MetricType.ACCURACY,
                    value=accuracy,
                    unit="ratio",
                    timestamp=end_time,
                    context={'ground_truth_available': True}
                )

            # Calculate consciousness detection performance
            if 'consciousness_assessment' in result:
                consciousness_performance = self._calculate_consciousness_performance(result)
                metrics[MetricType.CONSCIOUSNESS_DETECTION] = consciousness_performance

            # Calculate resource utilization
            final_resources = await self._capture_resource_usage()
            resource_utilization = self._calculate_resource_utilization(
                initial_resources, final_resources
            )
            metrics[MetricType.RESOURCE_UTILIZATION] = resource_utilization

            # Calculate temporal consistency
            temporal_consistency = self._calculate_temporal_consistency(result)
            metrics[MetricType.TEMPORAL_CONSISTENCY] = temporal_consistency

            # Update metric history
            for metric_type, metric in metrics.items():
                self.metric_history[metric_type].append(metric)

            # Update real-time statistics
            self._update_real_time_statistics(metrics)

            return metrics

        except Exception as e:
            # Record performance failure
            error_metric = PerformanceMetric(
                metric_type=MetricType.PROCESSING_LATENCY,
                value=float('inf'),
                unit="milliseconds",
                timestamp=time.time(),
                context={'error': str(e), 'failed': True}
            )
            return {MetricType.PROCESSING_LATENCY: error_metric}

    def _calculate_consciousness_performance(self, result: Dict) -> PerformanceMetric:
        """Calculate consciousness detection performance metrics."""
        consciousness_assessment = result.get('consciousness_assessment', {})

        # Accuracy of consciousness threshold detection
        predicted_conscious = consciousness_assessment.get('is_conscious', False)
        consciousness_strength = consciousness_assessment.get('consciousness_strength', 0.0)

        # Performance based on strength-threshold consistency
        threshold = 0.7
        threshold_consistent = (predicted_conscious and consciousness_strength >= threshold) or \
                              (not predicted_conscious and consciousness_strength < threshold)

        # Calculate detection confidence (how far from threshold)
        distance_from_threshold = abs(consciousness_strength - threshold)
        confidence_score = min(1.0, distance_from_threshold / 0.3)  # Normalize to 0-1

        # Overall consciousness detection performance
        performance_score = 0.7 * (1.0 if threshold_consistent else 0.0) + 0.3 * confidence_score

        return PerformanceMetric(
            metric_type=MetricType.CONSCIOUSNESS_DETECTION,
            value=performance_score,
            unit="ratio",
            timestamp=time.time(),
            context={
                'consciousness_strength': consciousness_strength,
                'threshold_consistent': threshold_consistent,
                'confidence_score': confidence_score
            }
        )

    def _calculate_temporal_consistency(self, result: Dict) -> PerformanceMetric:
        """Calculate temporal consistency of processing."""
        processing_stages = result.get('processing_stages', [])

        if len(processing_stages) < 2:
            return PerformanceMetric(
                metric_type=MetricType.TEMPORAL_CONSISTENCY,
                value=0.0,
                unit="ratio",
                timestamp=time.time(),
                context={'insufficient_stages': True}
            )

        # Calculate stage timing consistency
        stage_times = [stage.get('processing_time', 0.0) for stage in processing_stages]
        expected_times = [100, 150, 100, 100, 50]  # Expected times for each stage (ms)

        if len(stage_times) == len(expected_times):
            # Calculate timing deviations
            deviations = [abs(actual - expected) / expected
                         for actual, expected in zip(stage_times, expected_times)]
            consistency_score = max(0.0, 1.0 - np.mean(deviations))
        else:
            consistency_score = 0.5  # Partial consistency for incomplete stages

        return PerformanceMetric(
            metric_type=MetricType.TEMPORAL_CONSISTENCY,
            value=consistency_score,
            unit="ratio",
            timestamp=time.time(),
            context={
                'stage_times': stage_times,
                'expected_times': expected_times,
                'stage_count': len(processing_stages)
            }
        )

    async def measure_throughput_performance(self,
                                           processing_function,
                                           test_inputs: List[Dict],
                                           duration_seconds: float = 60.0) -> PerformanceMetric:
        """
        Measure system throughput over specified duration.

        Args:
            processing_function: Function to test
            test_inputs: List of test inputs to cycle through
            duration_seconds: Duration to run throughput test

        Returns:
            Throughput performance metric
        """
        start_time = time.time()
        end_time = start_time + duration_seconds
        completed_operations = 0
        failed_operations = 0
        input_index = 0

        while time.time() < end_time:
            try:
                # Get next test input (cycle through available inputs)
                test_input = test_inputs[input_index % len(test_inputs)]
                input_index += 1

                # Execute processing
                await processing_function(test_input)
                completed_operations += 1

            except Exception as e:
                failed_operations += 1
                logging.error(f"Throughput test operation failed: {e}")

            # Brief pause to prevent overwhelming the system
            await asyncio.sleep(0.001)

        actual_duration = time.time() - start_time
        throughput = completed_operations / actual_duration

        return PerformanceMetric(
            metric_type=MetricType.THROUGHPUT,
            value=throughput,
            unit="operations/second",
            timestamp=time.time(),
            context={
                'test_duration_seconds': actual_duration,
                'completed_operations': completed_operations,
                'failed_operations': failed_operations,
                'success_rate': completed_operations / (completed_operations + failed_operations) if (completed_operations + failed_operations) > 0 else 0.0
            }
        )

    async def measure_integration_performance(self,
                                            integration_functions: Dict[str, callable],
                                            test_data: Dict) -> Dict[str, PerformanceMetric]:
        """
        Measure performance of integrations with other consciousness forms.

        Args:
            integration_functions: Dictionary of integration functions to test
            test_data: Test data for integration calls

        Returns:
            Dictionary of integration performance metrics
        """
        integration_metrics = {}

        for integration_name, integration_function in integration_functions.items():
            start_time = time.time()

            try:
                # Execute integration with timing
                result = await integration_function(test_data)
                end_time = time.time()

                integration_latency = (end_time - start_time) * 1000  # Convert to ms

                # Assess integration quality
                integration_quality = self._assess_integration_quality(result)

                integration_metrics[integration_name] = PerformanceMetric(
                    metric_type=MetricType.INTEGRATION_PERFORMANCE,
                    value=integration_latency,
                    unit="milliseconds",
                    timestamp=end_time,
                    context={
                        'integration_name': integration_name,
                        'integration_quality': integration_quality,
                        'success': True
                    }
                )

            except Exception as e:
                integration_metrics[integration_name] = PerformanceMetric(
                    metric_type=MetricType.INTEGRATION_PERFORMANCE,
                    value=float('inf'),
                    unit="milliseconds",
                    timestamp=time.time(),
                    context={
                        'integration_name': integration_name,
                        'error': str(e),
                        'success': False
                    }
                )

        return integration_metrics
```

### Performance Analysis and Reporting

```python
class PerformanceAnalyzer:
    """
    Advanced performance analysis and reporting system.
    """

    def __init__(self, performance_monitor: RecurrentProcessingPerformanceMonitor):
        self.performance_monitor = performance_monitor
        self.analysis_cache = {}
        self.trend_analyzer = TrendAnalyzer()

    def calculate_performance_percentiles(self,
                                        metric_type: MetricType,
                                        percentiles: List[float] = [50, 75, 90, 95, 99]) -> Dict:
        """
        Calculate performance percentiles for specified metric.

        Args:
            metric_type: Type of metric to analyze
            percentiles: List of percentiles to calculate

        Returns:
            Dictionary of percentile values
        """
        if metric_type not in self.performance_monitor.metric_history:
            return {}

        metric_values = [
            metric.value for metric in self.performance_monitor.metric_history[metric_type]
            if not np.isinf(metric.value)  # Exclude failed operations
        ]

        if not metric_values:
            return {}

        percentile_results = {}
        for percentile in percentiles:
            percentile_results[f"p{percentile}"] = np.percentile(metric_values, percentile)

        return percentile_results

    def analyze_performance_trends(self,
                                 metric_type: MetricType,
                                 window_size: int = 100) -> Dict:
        """
        Analyze performance trends over time.

        Args:
            metric_type: Type of metric to analyze
            window_size: Size of moving window for trend analysis

        Returns:
            Trend analysis results
        """
        if metric_type not in self.performance_monitor.metric_history:
            return {'error': f'No data available for {metric_type.value}'}

        metrics = list(self.performance_monitor.metric_history[metric_type])
        if len(metrics) < window_size:
            return {'error': 'Insufficient data for trend analysis'}

        # Extract values and timestamps
        values = [m.value for m in metrics if not np.isinf(m.value)]
        timestamps = [m.timestamp for m in metrics if not np.isinf(m.value)]

        if len(values) < window_size:
            return {'error': 'Insufficient valid data points'}

        # Calculate trend using linear regression
        x = np.arange(len(values))
        trend_coeffs = np.polyfit(x, values, 1)
        trend_slope = trend_coeffs[0]

        # Calculate moving averages
        recent_avg = np.mean(values[-window_size//4:])
        baseline_avg = np.mean(values[:window_size//4])

        # Determine trend direction and strength
        trend_direction = 'improving' if trend_slope < 0 and metric_type in [MetricType.PROCESSING_LATENCY] \
                         else 'improving' if trend_slope > 0 and metric_type in [MetricType.THROUGHPUT, MetricType.ACCURACY] \
                         else 'declining' if abs(trend_slope) > 0.01 else 'stable'

        return {
            'trend_direction': trend_direction,
            'trend_slope': trend_slope,
            'trend_strength': abs(trend_slope),
            'recent_performance': recent_avg,
            'baseline_performance': baseline_avg,
            'improvement_ratio': (recent_avg / baseline_avg) if baseline_avg != 0 else 1.0,
            'data_points_analyzed': len(values)
        }

    def identify_performance_anomalies(self,
                                     metric_type: MetricType,
                                     anomaly_threshold: float = 2.0) -> List[Dict]:
        """
        Identify performance anomalies using statistical analysis.

        Args:
            metric_type: Type of metric to analyze
            anomaly_threshold: Standard deviation threshold for anomaly detection

        Returns:
            List of identified anomalies
        """
        if metric_type not in self.performance_monitor.metric_history:
            return []

        metrics = list(self.performance_monitor.metric_history[metric_type])
        values = [m.value for m in metrics if not np.isinf(m.value)]

        if len(values) < 50:  # Need sufficient data for anomaly detection
            return []

        # Calculate statistical thresholds
        mean_value = np.mean(values)
        std_value = np.std(values)
        upper_threshold = mean_value + (anomaly_threshold * std_value)
        lower_threshold = mean_value - (anomaly_threshold * std_value)

        anomalies = []
        for metric in metrics:
            if not np.isinf(metric.value):
                if metric.value > upper_threshold or metric.value < lower_threshold:
                    anomaly_type = 'high' if metric.value > upper_threshold else 'low'
                    severity = abs(metric.value - mean_value) / std_value

                    anomalies.append({
                        'timestamp': metric.timestamp,
                        'value': metric.value,
                        'mean_value': mean_value,
                        'anomaly_type': anomaly_type,
                        'severity': severity,
                        'context': metric.context
                    })

        return sorted(anomalies, key=lambda x: x['severity'], reverse=True)

    def generate_performance_report(self,
                                  timeframe_hours: float = 24.0,
                                  include_trends: bool = True,
                                  include_anomalies: bool = True) -> Dict:
        """
        Generate comprehensive performance report.

        Args:
            timeframe_hours: Time period to analyze
            include_trends: Whether to include trend analysis
            include_anomalies: Whether to include anomaly detection

        Returns:
            Comprehensive performance report
        """
        report = {
            'report_timestamp': time.time(),
            'timeframe_hours': timeframe_hours,
            'metric_summaries': {},
            'overall_performance': {},
            'recommendations': []
        }

        # Analyze each metric type
        for metric_type in MetricType:
            metric_summary = {
                'metric_type': metric_type.value,
                'benchmark': self.performance_monitor.benchmarks.get(metric_type),
                'current_statistics': {},
                'performance_level': PerformanceLevel.POOR
            }

            # Calculate current statistics
            percentiles = self.calculate_performance_percentiles(metric_type)
            if percentiles:
                metric_summary['current_statistics'] = percentiles

                # Determine performance level based on median (p50)
                current_median = percentiles.get('p50', 0)
                benchmark = self.performance_monitor.benchmarks.get(metric_type)
                if benchmark:
                    metric_summary['performance_level'] = self._determine_performance_level(
                        current_median, benchmark
                    )

            # Add trend analysis if requested
            if include_trends:
                trend_analysis = self.analyze_performance_trends(metric_type)
                metric_summary['trend_analysis'] = trend_analysis

            # Add anomaly detection if requested
            if include_anomalies:
                anomalies = self.identify_performance_anomalies(metric_type)
                metric_summary['recent_anomalies'] = len(anomalies)
                metric_summary['critical_anomalies'] = [
                    a for a in anomalies if a['severity'] > 3.0
                ]

            report['metric_summaries'][metric_type.value] = metric_summary

        # Calculate overall performance score
        report['overall_performance'] = self._calculate_overall_performance_score(
            report['metric_summaries']
        )

        # Generate recommendations
        report['recommendations'] = self._generate_performance_recommendations(
            report['metric_summaries']
        )

        return report

    def _determine_performance_level(self,
                                   current_value: float,
                                   benchmark: PerformanceBenchmark) -> PerformanceLevel:
        """Determine performance level based on benchmark comparison."""

        # For latency metrics, lower is better
        if benchmark.metric_type in [MetricType.PROCESSING_LATENCY, MetricType.INTEGRATION_PERFORMANCE]:
            if current_value <= benchmark.target_value:
                return PerformanceLevel.OPTIMAL
            elif current_value <= benchmark.threshold_excellent:
                return PerformanceLevel.EXCELLENT
            elif current_value <= benchmark.threshold_good:
                return PerformanceLevel.GOOD
            elif current_value <= benchmark.threshold_acceptable:
                return PerformanceLevel.ACCEPTABLE
            else:
                return PerformanceLevel.POOR

        # For other metrics, higher is better
        else:
            if current_value >= benchmark.target_value:
                return PerformanceLevel.OPTIMAL
            elif current_value >= benchmark.threshold_excellent:
                return PerformanceLevel.EXCELLENT
            elif current_value >= benchmark.threshold_good:
                return PerformanceLevel.GOOD
            elif current_value >= benchmark.threshold_acceptable:
                return PerformanceLevel.ACCEPTABLE
            else:
                return PerformanceLevel.POOR

    def _calculate_overall_performance_score(self, metric_summaries: Dict) -> Dict:
        """Calculate overall system performance score."""
        performance_levels = []
        weights = {
            MetricType.PROCESSING_LATENCY.value: 0.25,
            MetricType.THROUGHPUT.value: 0.20,
            MetricType.ACCURACY.value: 0.25,
            MetricType.CONSCIOUSNESS_DETECTION.value: 0.15,
            MetricType.RESOURCE_UTILIZATION.value: 0.10,
            MetricType.TEMPORAL_CONSISTENCY.value: 0.05
        }

        weighted_score = 0.0
        total_weight = 0.0

        level_scores = {
            PerformanceLevel.OPTIMAL: 1.0,
            PerformanceLevel.EXCELLENT: 0.9,
            PerformanceLevel.GOOD: 0.75,
            PerformanceLevel.ACCEPTABLE: 0.6,
            PerformanceLevel.POOR: 0.3
        }

        for metric_name, summary in metric_summaries.items():
            if metric_name in weights:
                weight = weights[metric_name]
                level = summary.get('performance_level', PerformanceLevel.POOR)
                score = level_scores[level]

                weighted_score += weight * score
                total_weight += weight

        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # Determine overall performance level
        if overall_score >= 0.95:
            overall_level = PerformanceLevel.OPTIMAL
        elif overall_score >= 0.85:
            overall_level = PerformanceLevel.EXCELLENT
        elif overall_score >= 0.70:
            overall_level = PerformanceLevel.GOOD
        elif overall_score >= 0.55:
            overall_level = PerformanceLevel.ACCEPTABLE
        else:
            overall_level = PerformanceLevel.POOR

        return {
            'overall_score': overall_score,
            'overall_level': overall_level.value,
            'component_scores': {
                metric: summary.get('performance_level', PerformanceLevel.POOR).value
                for metric, summary in metric_summaries.items()
            }
        }
```

This performance metrics system provides comprehensive measurement, analysis, and reporting capabilities for the recurrent processing implementation, enabling continuous performance optimization and quality assurance.