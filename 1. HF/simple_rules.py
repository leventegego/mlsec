import numpy as np

class AttackPlausibilityChecker:
    """
    Rule-based checker for adversarial attack plausibility.
    Uses domain knowledge constraints from NSL-KDD.
    """
    
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.constraints = self._define_constraints()
    
    def _define_constraints(self):
        """Define hard constraints that valid attacks must satisfy."""
        return {
            # Core attack characteristics that CANNOT be violated
            "critical_features": {
                "DoS": {
                    "count": {"min": 400, "max": 511, "reason": "DoS needs high connection rate"},
                    "srv_count": {"min": 400, "max": 511, "reason": "Service flooding"},
                    "serror_rate": {"min": 0.9, "max": 1.0, "reason": "SYN flood signature (for neptune)"}
                },
                "Probe": {
                    "count": {"min": 100, "max": 511, "reason": "Scanning needs many connections"},
                    "rerror_rate": {"min": 0.7, "max": 1.0, "reason": "Probing closed ports"}
                },
                "R2L": {
                    # More permissive - these attacks vary more
                },
                "U2R": {
                    "root_shell": {"min": 0, "max": 1, "reason": "Binary - either got root or not"},
                    "num_root": {"min": 0, "max": 100, "reason": "Root operations"}
                }
            },
            
            # Relationships between features that must hold
            "feature_relationships": [
                {
                    "name": "duration_bytes_consistency",
                    "check": lambda x: self._check_duration_bytes(x),
                    "reason": "Duration=0 should mean low bytes"
                },
                {
                    "name": "serror_count_consistency", 
                    "check": lambda x: self._check_serror_count(x),
                    "reason": "High serror_rate needs high count"
                },
                {
                    "name": "logged_in_consistency",
                    "check": lambda x: self._check_logged_in_features(x),
                    "reason": "If logged_in=0, can't have shells/file_creations"
                }
            ],
            
            # Physical/logical bounds
            "absolute_bounds": {
                "duration": {"min": 0, "max": 86400, "reason": "Can't be negative or > 1 day"},
                "src_bytes": {"min": 0, "max": 1e9, "reason": "Can't be negative"},
                "dst_bytes": {"min": 0, "max": 1e9, "reason": "Can't be negative"},
                "count": {"min": 0, "max": 511, "reason": "NSL-KDD max"},
                "srv_count": {"min": 0, "max": 511, "reason": "NSL-KDD max"},
                "serror_rate": {"min": 0.0, "max": 1.0, "reason": "Rate must be [0,1]"},
                "rerror_rate": {"min": 0.0, "max": 1.0, "reason": "Rate must be [0,1]"},
                "same_srv_rate": {"min": 0.0, "max": 1.0, "reason": "Rate must be [0,1]"},
                "diff_srv_rate": {"min": 0.0, "max": 1.0, "reason": "Rate must be [0,1]"}
            }
        }
    
    def _check_duration_bytes(self, x):
        """Check duration-bytes consistency."""
        duration_idx = self._get_feature_index("duration")
        src_bytes_idx = self._get_feature_index("src_bytes")
        
        if duration_idx is None or src_bytes_idx is None:
            return True, "OK"
        
        duration = x[duration_idx]
        src_bytes = x[src_bytes_idx]
        
        # Case 1: Very short duration should mean low bytes
        if duration < 1 and src_bytes > 1000:
            return False, "Duration~0 but high src_bytes is suspicious"
        
        # Case 2: Long duration should mean some data transfer
        if duration > 100 and src_bytes < 10:
            return False, "Long duration but no data transfer is suspicious"
        
        # Case 3: Very long duration with zero bytes is highly suspicious
        if duration > 300 and src_bytes == 0:
            return False, "Very long connection with zero bytes transferred"
        
        return True, "OK"
    
    def _check_serror_count(self, x):
        """High serror_rate needs sufficient count."""
        serror_rate_idx = self._get_feature_index("serror_rate")
        count_idx = self._get_feature_index("count")
        
        if x[serror_rate_idx] > 0.9 and x[count_idx] < 50:
            return False, "High serror_rate but low count - SYN flood needs high rate"
        return True, "OK"
    
    def _check_logged_in_features(self, x):
        """If not logged in, can't have shells/file operations."""
        logged_in_idx = self._get_feature_index("logged_in")
        num_shells_idx = self._get_feature_index("num_shells")
        num_file_creations_idx = self._get_feature_index("num_file_creations")
        
        if x[logged_in_idx] < 0.5:  # Not logged in
            if x[num_shells_idx] > 0 or x[num_file_creations_idx] > 0:
                return False, "Not logged in but has shells/file_creations"
        return True, "OK"
    
    def _get_feature_index(self, feature_name):
        """Get index of feature in array."""
        # Handle one-hot encoded features
        for i, name in enumerate(self.feature_names):
            if name == feature_name or name.startswith(feature_name + "_"):
                return i
        return None
    
    def _round_features(self, x):
        """Round features to appropriate precision."""
        x_rounded = x.copy()
        for i, name in enumerate(self.feature_names):
            if 'rate' in name:  # Rates: 4 decimals
                x_rounded[i] = np.round(x[i], 4)
            elif any(count in name for count in ['count', 'num_']):  # Counts: whole numbers
                x_rounded[i] = np.round(x[i], 0)
            else:  # Other continuous: 2 decimals
                x_rounded[i] = np.round(x[i], 2)
        return x_rounded
        
    def check_plausibility(self, x_adversarial, attack_type="DoS", 
                          scaler=None, verbose=False):
        """
        Check if adversarial example is plausible.
        
        Args:
            x_adversarial: Adversarial example (standardized)
            attack_type: Type of attack (DoS, Probe, R2L, U2R)
            scaler: StandardScaler to inverse transform
            verbose: Print detailed results
        
        Returns:
            is_plausible: Boolean
            violations: List of constraint violations
            plausibility_score: Float [0, 1]
        """
        violations = []
        
        # Inverse transform to original scale for checking
        if scaler is not None:
            x_adv_orig = scaler.inverse_transform(x_adversarial.reshape(1, -1))[0]
        else:
            x_adv_orig = x_adversarial

        x_adv_orig = self._round_features(x_adv_orig)
        
        # Check 1: Critical attack features
        if attack_type in self.constraints["critical_features"]:
            critical = self.constraints["critical_features"][attack_type]
            for feature, bounds in critical.items():
                idx = self._get_feature_index(feature)
                if idx is not None:
                    value = x_adv_orig[idx]
                    if not (bounds["min"] <= value <= bounds["max"]):
                        violations.append({
                            "type": "critical_feature",
                            "feature": feature,
                            "value": value,
                            "expected": f"[{bounds['min']}, {bounds['max']}]",
                            "reason": bounds["reason"]
                        })
        
        # Check 2: Feature relationships
        for relationship in self.constraints["feature_relationships"]:
            is_valid, reason = relationship["check"](x_adv_orig)
            if not is_valid:
                violations.append({
                    "type": "relationship",
                    "name": relationship["name"],
                    "reason": reason
                })
        
        # Check 3: Absolute bounds
        for feature, bounds in self.constraints["absolute_bounds"].items():
            idx = self._get_feature_index(feature)
            if idx is not None:
                value = x_adv_orig[idx]
                if not (bounds["min"] <= value <= bounds["max"]):
                    violations.append({
                        "type": "absolute_bound",
                        "feature": feature,
                        "value": value,
                        "expected": f"[{bounds['min']}, {bounds['max']}]",
                        "reason": bounds["reason"]
                    })
        
        # Calculate plausibility score
        is_plausible = len(violations) == 0
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"PLAUSIBILITY CHECK: {attack_type} Attack")
            print(f"{'='*70}")
            print(f"Is Plausible: {is_plausible}")
            print(f"Violations: {len(violations)}")
            
            if violations:
                print(f"\nViolation Details:")
                for v in violations:
                    print(f"  • {v['type']}: {v.get('feature', v.get('name'))}")
                    print(f"    Reason: {v['reason']}")
                    if 'value' in v:
                        print(f"    Got {v['value']:.4f}, Expected {v['expected']}")
        
        return is_plausible, violations

