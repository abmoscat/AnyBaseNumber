import math
import numpy as np
from typing import Union, List, Optional, Callable
from functools import lru_cache

class MLOptimizedPhinaryNumber:
    """
    Machine Learning optimized base-φ (phinary) number system.
    Designed for numerical optimization algorithms like Nelder-Mead.
    
    Key ML optimizations:
    - Vectorized operations for batch processing
    - Gradient-friendly conversions
    - Numerical stability for optimization
    - Integration with NumPy ecosystem
    """
    
    # High-precision constants for ML stability
    PHI = (1 + math.sqrt(5)) / 2
    PHI_INV = (math.sqrt(5) - 1) / 2
    LOG_PHI = math.log(PHI)
    
    # Precomputed tables with extended precision
    _PHI_POWERS = None
    _PHI_LOG_POWERS = None
    _CONVERSION_CACHE = {}
    _MAX_PRECOMPUTED = 128
    
    @classmethod
    def _init_tables(cls):
        """Initialize high-precision lookup tables for ML operations."""
        if cls._PHI_POWERS is None:
            # Use higher precision for ML applications
            cls._PHI_POWERS = np.array([cls.PHI ** i for i in range(-64, cls._MAX_PRECOMPUTED)])
            cls._PHI_LOG_POWERS = np.log(cls._PHI_POWERS)
    
    def __init__(self, value: Union[int, float, str, np.ndarray, 'MLOptimizedPhinaryNumber'] = 0, 
                 precision_bits: int = 53):
        """
        Initialize with ML-friendly features.
        
        Args:
            value: Input value (supports vectorized inputs)
            precision_bits: Precision for ML operations (default: double precision)
        """
        self._init_tables()
        self.precision_bits = precision_bits
        
        if isinstance(value, MLOptimizedPhinaryNumber):
            self.bits = value.bits
            self._decimal_cache = getattr(value, '_decimal_cache', None)
        elif isinstance(value, np.ndarray):
            # Vectorized initialization for batch ML operations
            self.bits = self._vectorized_from_decimal(value)
            self._decimal_cache = None
        elif isinstance(value, str):
            self.bits = self._parse_string_ml(value)
            self._decimal_cache = None
        else:
            self.bits = self._from_decimal_ml_optimized(float(value))
            self._decimal_cache = float(value) if isinstance(value, (int, float)) else None
        
        self._normalize_ml_safe()
    
    @lru_cache(maxsize=10000)
    def _cached_decimal_conversion(self, bits: int) -> float:
        """LRU cached decimal conversion for repeated ML evaluations."""
        return self._bits_to_decimal_fast(bits)
    
    def _from_decimal_ml_optimized(self, value: float) -> int:
        """
        ML-optimized decimal to phinary conversion.
        Focuses on numerical stability and gradient preservation.
        """
        if abs(value) < 1e-15:
            return 0
        
        # Use cached conversion for common values
        if value in self._CONVERSION_CACHE:
            return self._CONVERSION_CACHE[value]
        
        bits = 0
        remaining = abs(value)
        tolerance = 1e-14  # Higher precision for ML
        
        # Optimized greedy algorithm with stability checks
        for i in range(127, -65, -1):  # Extended range for ML
            phi_power = self._PHI_POWERS[i + 64]  # Offset for negative indices
            if remaining >= phi_power - tolerance:
                bits |= (1 << (i + 64))
                remaining -= phi_power
                if remaining < tolerance:
                    break
        
        # Cache common conversions
        if len(self._CONVERSION_CACHE) < 1000:
            self._CONVERSION_CACHE[value] = bits
        
        return bits
    
    def _vectorized_from_decimal(self, values: np.ndarray) -> np.ndarray:
        """Vectorized conversion for batch ML operations."""
        vectorized_convert = np.vectorize(self._from_decimal_ml_optimized)
        return vectorized_convert(values)
    
    def _normalize_ml_safe(self):
        """
        ML-safe normalization that preserves numerical properties
        important for optimization algorithms.
        """
        if self.bits == 0:
            return
        
        max_iterations = 100  # Prevent infinite loops in ML context
        iteration = 0
        
        while iteration < max_iterations:
            consecutive = self.bits & (self.bits >> 1)
            if consecutive == 0:
                break
            
            # More conservative normalization for ML stability
            self.bits ^= consecutive | (consecutive << 1)
            self.bits |= (consecutive << 2)
            iteration += 1
        
        # Clear decimal cache after normalization
        self._decimal_cache = None
    
    def _bits_to_decimal_fast(self, bits: int) -> float:
        """Fast bit-to-decimal conversion using precomputed powers."""
        if bits == 0:
            return 0.0
        
        result = 0.0
        bit_pos = 0
        temp_bits = bits
        
        while temp_bits and bit_pos < 192:  # Extended range
            if temp_bits & 1:
                if bit_pos < len(self._PHI_POWERS):
                    result += self._PHI_POWERS[bit_pos]
                else:
                    # Fallback for very large numbers
                    result += self.PHI ** (bit_pos - 64)
            temp_bits >>= 1
            bit_pos += 1
        
        return result
    
    def to_decimal(self) -> float:
        """Cached decimal conversion optimized for ML repeated evaluations."""
        if self._decimal_cache is not None:
            return self._decimal_cache
        
        decimal_val = self._cached_decimal_conversion(self.bits)
        self._decimal_cache = decimal_val
        return decimal_val
    
    def to_numpy(self) -> np.float64:
        """Convert to NumPy float64 for ML integration."""
        return np.float64(self.to_decimal())
    
    # ML-optimized arithmetic operations
    def __add__(self, other: Union['MLOptimizedPhinaryNumber', float, int]) -> 'MLOptimizedPhinaryNumber':
        """ML-optimized addition with numerical stability."""
        if not isinstance(other, MLOptimizedPhinaryNumber):
            other = MLOptimizedPhinaryNumber(other)
        
        result = MLOptimizedPhinaryNumber.__new__(MLOptimizedPhinaryNumber)
        result._init_tables()
        result.precision_bits = max(self.precision_bits, other.precision_bits)
        
        # Fast bit-level addition
        a, b = self.bits, other.bits
        result.bits = a ^ b
        carry = (a & b) << 1
        
        # Optimized carry propagation for ML stability
        max_carries = 64  # Limit for ML applications
        carry_count = 0
        
        while carry and carry_count < max_carries:
            temp = result.bits ^ carry
            carry = (result.bits & carry) << 1
            result.bits = temp
            carry_count += 1
        
        result._normalize_ml_safe()
        return result
    
    def __sub__(self, other: Union['MLOptimizedPhinaryNumber', float, int]) -> 'MLOptimizedPhinaryNumber':
        """ML-optimized subtraction with gradient preservation."""
        if not isinstance(other, MLOptimizedPhinaryNumber):
            other = MLOptimizedPhinaryNumber(other)
        
        # For ML applications, convert to decimal for stability
        decimal_result = self.to_decimal() - other.to_decimal()
        return MLOptimizedPhinaryNumber(decimal_result)
    
    def __mul__(self, other: Union['MLOptimizedPhinaryNumber', float, int]) -> 'MLOptimizedPhinaryNumber':
        """ML-optimized multiplication preserving gradient information."""
        if isinstance(other, (int, float)):
            # Scalar multiplication - very common in ML
            if other == 0:
                return MLOptimizedPhinaryNumber(0)
            decimal_result = self.to_decimal() * other
            return MLOptimizedPhinaryNumber(decimal_result)
        
        if not isinstance(other, MLOptimizedPhinaryNumber):
            other = MLOptimizedPhinaryNumber(other)
        
        # For ML, use decimal multiplication for stability
        decimal_result = self.to_decimal() * other.to_decimal()
        return MLOptimizedPhinaryNumber(decimal_result)
    
    def __truediv__(self, other: Union['MLOptimizedPhinaryNumber', float, int]) -> 'MLOptimizedPhinaryNumber':
        """ML-optimized division with numerical stability."""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            decimal_result = self.to_decimal() / other
            return MLOptimizedPhinaryNumber(decimal_result)
        
        if not isinstance(other, MLOptimizedPhinaryNumber):
            other = MLOptimizedPhinaryNumber(other)
        
        other_decimal = other.to_decimal()
        if abs(other_decimal) < 1e-15:
            raise ZeroDivisionError("Division by zero")
        
        decimal_result = self.to_decimal() / other_decimal
        return MLOptimizedPhinaryNumber(decimal_result)
    
    def __pow__(self, exponent: Union[float, int]) -> 'MLOptimizedPhinaryNumber':
        """Power operation optimized for ML gradient computation."""
        base_decimal = self.to_decimal()
        if base_decimal <= 0 and not isinstance(exponent, int):
            raise ValueError("Cannot raise non-positive number to fractional power")
        
        result_decimal = base_decimal ** exponent
        return MLOptimizedPhinaryNumber(result_decimal)
    
    # ML-specific methods
    def gradient_safe_log(self) -> float:
        """Numerically stable logarithm for gradient computation."""
        decimal_val = self.to_decimal()
        if decimal_val <= 0:
            return float('-inf')
        return math.log(decimal_val)
    
    def gradient_safe_exp(self) -> 'MLOptimizedPhinaryNumber':
        """Numerically stable exponential for gradient computation."""
        decimal_val = self.to_decimal()
        if decimal_val > 700:  # Prevent overflow
            return MLOptimizedPhinaryNumber(float('inf'))
        return MLOptimizedPhinaryNumber(math.exp(decimal_val))
    
    def clamp(self, min_val: float = -1e10, max_val: float = 1e10) -> 'MLOptimizedPhinaryNumber':
        """Clamp values for ML numerical stability."""
        decimal_val = self.to_decimal()
        clamped = max(min_val, min(max_val, decimal_val))
        return MLOptimizedPhinaryNumber(clamped)
    
    # Nelder-Mead specific optimizations
    def reflect(self, centroid: 'MLOptimizedPhinaryNumber', alpha: float = 1.0) -> 'MLOptimizedPhinaryNumber':
        """Reflection operation for Nelder-Mead."""
        # xr = centroid + alpha * (centroid - self)
        return centroid + MLOptimizedPhinaryNumber(alpha) * (centroid - self)
    
    def expand(self, centroid: 'MLOptimizedPhinaryNumber', gamma: float = 2.0) -> 'MLOptimizedPhinaryNumber':
        """Expansion operation for Nelder-Mead."""
        # xe = centroid + gamma * (self - centroid)
        return centroid + MLOptimizedPhinaryNumber(gamma) * (self - centroid)
    
    def contract(self, centroid: 'MLOptimizedPhinaryNumber', rho: float = 0.5) -> 'MLOptimizedPhinaryNumber':
        """Contraction operation for Nelder-Mead."""
        # xc = centroid + rho * (self - centroid)
        return centroid + MLOptimizedPhinaryNumber(rho) * (self - centroid)
    
    # Comparison operations for optimization
    def __lt__(self, other: Union['MLOptimizedPhinaryNumber', float, int]) -> bool:
        if isinstance(other, (int, float)):
            return self.to_decimal() < other
        return self.to_decimal() < other.to_decimal()
    
    def __le__(self, other: Union['MLOptimizedPhinaryNumber', float, int]) -> bool:
        if isinstance(other, (int, float)):
            return self.to_decimal() <= other
        return self.to_decimal() <= other.to_decimal()
    
    def __gt__(self, other: Union['MLOptimizedPhinaryNumber', float, int]) -> bool:
        if isinstance(other, (int, float)):
            return self.to_decimal() > other
        return self.to_decimal() > other.to_decimal()
    
    def __ge__(self, other: Union['MLOptimizedPhinaryNumber', float, int]) -> bool:
        if isinstance(other, (int, float)):
            return self.to_decimal() >= other
        return self.to_decimal() >= other.to_decimal()
    
    def __eq__(self, other: Union['MLOptimizedPhinaryNumber', float, int]) -> bool:
        if isinstance(other, (int, float)):
            return abs(self.to_decimal() - other) < 1e-12
        return abs(self.to_decimal() - other.to_decimal()) < 1e-12
    
    # NumPy integration
    def __array__(self) -> np.ndarray:
        """NumPy array protocol support."""
        return np.array(self.to_decimal())
    
    def __float__(self) -> float:
        """Python float conversion."""
        return self.to_decimal()
    
    def __repr__(self) -> str:
        return f"MLPhinary({self.to_decimal():.6f})"
    
    def __str__(self) -> str:
        return f"{self.to_decimal():.6f}φ"


# Nelder-Mead integration utilities
class NelderMeadPhinary:
    """
    Nelder-Mead optimizer using phinary arithmetic.
    Optimized for the unique properties of base-φ representation.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 rho: float = 0.5, sigma: float = 0.5):
        self.alpha = alpha  # Reflection coefficient
        self.gamma = gamma  # Expansion coefficient  
        self.rho = rho     # Contraction coefficient
        self.sigma = sigma # Shrink coefficient
    
    def optimize(self, objective_func: Callable, 
                 initial_simplex: List[List[float]], 
                 max_iterations: int = 1000,
                 tolerance: float = 1e-8) -> tuple:
        """
        Nelder-Mead optimization using phinary arithmetic.
        
        Returns:
            (best_point, best_value, iterations)
        """
        # Convert initial simplex to phinary
        simplex = []
        for point in initial_simplex:
            phinary_point = [MLOptimizedPhinaryNumber(x) for x in point]
            simplex.append(phinary_point)
        
        # Evaluate initial simplex
        values = [objective_func([x.to_decimal() for x in point]) for point in simplex]
        
        for iteration in range(max_iterations):
            # Sort simplex by function values
            sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
            simplex = [simplex[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]
            
            # Check convergence
            if abs(values[-1] - values[0]) < tolerance:
                break
            
            # Calculate centroid (excluding worst point)
            n = len(simplex[0])
            centroid = []
            for dim in range(n):
                coord_sum = MLOptimizedPhinaryNumber(0)
                for i in range(len(simplex) - 1):
                    coord_sum = coord_sum + simplex[i][dim]
                centroid.append(coord_sum / (len(simplex) - 1))
            
            # Reflection
            worst_point = simplex[-1]
            reflected_point = []
            for dim in range(n):
                reflected = worst_point[dim].reflect(centroid[dim], self.alpha)
                reflected_point.append(reflected)
            
            reflected_value = objective_func([x.to_decimal() for x in reflected_point])
            
            if values[0] <= reflected_value < values[-2]:
                # Accept reflection
                simplex[-1] = reflected_point
                values[-1] = reflected_value
            elif reflected_value < values[0]:
                # Try expansion
                expanded_point = []
                for dim in range(n):
                    expanded = reflected_point[dim].expand(centroid[dim], self.gamma)
                    expanded_point.append(expanded)
                
                expanded_value = objective_func([x.to_decimal() for x in expanded_point])
                
                if expanded_value < reflected_value:
                    simplex[-1] = expanded_point
                    values[-1] = expanded_value
                else:
                    simplex[-1] = reflected_point
                    values[-1] = reflected_value
            else:
                # Try contraction
                if reflected_value < values[-1]:
                    # Outside contraction
                    contracted_point = []
                    for dim in range(n):
                        contracted = reflected_point[dim].contract(centroid[dim], self.rho)
                        contracted_point.append(contracted)
                else:
                    # Inside contraction
                    contracted_point = []
                    for dim in range(n):
                        contracted = worst_point[dim].contract(centroid[dim], self.rho)
                        contracted_point.append(contracted)
                
                contracted_value = objective_func([x.to_decimal() for x in contracted_point])
                
                if contracted_value < min(reflected_value, values[-1]):
                    simplex[-1] = contracted_point
                    values[-1] = contracted_value
                else:
                    # Shrink simplex
                    best_point = simplex[0]
                    for i in range(1, len(simplex)):
                        for dim in range(n):
                            new_coord = best_point[dim] + MLOptimizedPhinaryNumber(self.sigma) * (simplex[i][dim] - best_point[dim])
                            simplex[i][dim] = new_coord
                        values[i] = objective_func([x.to_decimal() for x in simplex[i]])
        
        # Return best result
        best_point = [x.to_decimal() for x in simplex[0]]
        return best_point, values[0], iteration + 1


# Example usage and testing
if __name__ == "__main__":
    import time
    
    print("=== ML-Optimized Phinary Number System ===\n")
    
    # Test basic ML operations
    a = MLOptimizedPhinaryNumber(1.5)
    b = MLOptimizedPhinaryNumber(2.618)  # ≈ φ²
    
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"a / b = {a / b}")
    
    # Test Nelder-Mead specific operations
    centroid = MLOptimizedPhinaryNumber(3.0)
    reflected = a.reflect(centroid, 1.0)
    expanded = reflected.expand(centroid, 2.0)
    
    print(f"\nNelder-Mead operations:")
    print(f"Reflected: {reflected}")
    print(f"Expanded: {expanded}")
    
    # Performance test for ML applications
    print(f"\n=== ML Performance Test ===")
    
    test_data = [MLOptimizedPhinaryNumber(i * 0.1) for i in range(1000)]
    
    start_time = time.time()
    for _ in range(100):
        for i in range(len(test_data) - 1):
            result = test_data[i] * 2.0 + test_data[i + 1] / 3.0
    ml_time = time.time() - start_time
    
    print(f"ML operations: {ml_time:.4f} seconds (99,900 mixed operations)")
    print(f"Operations per second: {99900 / ml_time:.0f}")
    
    # Test Nelder-Mead optimizer
    print(f"\n=== Nelder-Mead Test ===")
    
    # Optimize simple quadratic function: (x-2)² + (y-3)²
    def objective(point):
        x, y = point
        return (x - 2)**2 + (y - 3)**2
    
    # Initial simplex
    initial_simplex = [[0, 0], [1, 0], [0, 1]]
    
    optimizer = NelderMeadPhinary()
    best_point, best_value, iterations = optimizer.optimize(
        objective, initial_simplex, max_iterations=100
    )
    
    print(f"Optimization result:")
    print(f"Best point: ({best_point[0]:.6f}, {best_point[1]:.6f})")
    print(f"Best value: {best_value:.6f}")
    print(f"Iterations: {iterations}")
    print(f"Expected: (2.0, 3.0) with value 0.0")