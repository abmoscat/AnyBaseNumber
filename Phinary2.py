import math
from typing import List, Union, Set
import time
class PhinaryNumber:
    """
    Represents a number in base-φ (phinary) representation.
    Uses the Zeckendorf representation (no consecutive 1s).
    """
    
    PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ≈ 1.618
    
    def __init__(self, coefficients: Union[List[int], str, float] = None):
        """
        Initialize a phinary number.
        
        Args:
            coefficients: Can be:
                - List of coefficients [d_k, d_{k-1}, ..., d_0] where d_k ∈ {0,1}
                - String like "101" representing the binary-like coefficients
                - Float/int to convert from decimal
        """
        if coefficients is None:
            self.coeffs = [0]
        elif isinstance(coefficients, (list, tuple)):
            self.coeffs = list(coefficients)
        elif isinstance(coefficients, str):
            # Parse string like "101" -> [1, 0, 1]
            self.coeffs = [int(d) for d in coefficients]
        elif isinstance(coefficients, (int, float)):
            # Convert from decimal
            # self.coeffs = self._from_decimal(float(coefficients))
            raise NotImplementedError("From decimal not implemented yet")

        else:
            raise ValueError("Invalid input type for coefficients")
        
        self._normalize()

    def _normalize(self):
        """Apply the Zeckendorf representation (no consecutive 1s)."""
        if not self.coeffs:
            self.coeffs = [0]
            return
            
        # Remove leading zeros
        while len(self.coeffs) > 1 and self.coeffs[0] == 0:
            self.coeffs.pop(0)
        
        # Apply the golden ratio identity: φ^n = φ^(n-1) + φ^(n-2)
        # This means if we have consecutive 1s, we can combine them
        changed = True
        while changed:
            changed = False
            
            # Check for consecutive 1s from right to left
            for i in range(len(self.coeffs) - 1, 0, -1):
                if self.coeffs[i] == 1 and self.coeffs[i-1] == 1:
                    # Replace φ^(k-1) + φ^k with φ^(k+1)
                    self.coeffs[i] = 0
                    self.coeffs[i-1] = 0
                    
                    # Add to position k+1 (or create it)
                    if i-2 >= 0:
                        self.coeffs[i-2] += 1
                    else:
                        self.coeffs.insert(0, 1)
                    
                    changed = True
                    break
            
            # Handle carries (when coefficient > 1)
            for i in range(len(self.coeffs) - 1, -1, -1):
                if self.coeffs[i] > 1:
                    carry = self.coeffs[i] // 2
                    self.coeffs[i] %= 2
                    
                    if carry > 0:
                        if i == 0:
                            self.coeffs.insert(0, carry)
                        else:
                            self.coeffs[i-1] += carry
                        changed = True
        
        # Remove leading zeros again
        while len(self.coeffs) > 1 and self.coeffs[0] == 0:
            self.coeffs.pop(0)
    
    def __add__(self, other: 'PhinaryNumber') -> 'PhinaryNumber':
        """Add two phinary numbers."""
        if not isinstance(other, PhinaryNumber):
            other = PhinaryNumber(other)
        
        # Make both numbers the same length by padding with zeros
        max_len = max(len(self.coeffs), len(other.coeffs))
        
        # Pad from the left (higher powers)
        self_padded = [0] * (max_len - len(self.coeffs)) + self.coeffs
        other_padded = [0] * (max_len - len(other.coeffs)) + other.coeffs
        
        # Add coefficient by coefficient
        result_coeffs = []
        for i in range(max_len):
            result_coeffs.append(self_padded[i] + other_padded[i])
        
        # Create result and normalize
        result = PhinaryNumber(result_coeffs)
        return result
    
    def __str__(self) -> str:
        """String representation showing coefficients."""
        return ''.join(map(str, self.coeffs)) + 'φ'
    
    def __repr__(self) -> str:
        return f"PhinaryNumber('{self}'[≈{self.to_decimal():.6f}])"
    
    def detailed_repr(self) -> str:
        """Show the detailed mathematical representation."""
        terms = []
        n = len(self.coeffs)
        for i, coeff in enumerate(self.coeffs):
            if coeff != 0:
                power = n - 1 - i
                if power == 0:
                    terms.append(f"{coeff}")
                elif power == 1:
                    terms.append(f"{coeff}φ")
                else:
                    terms.append(f"{coeff}φ^{power}")
        
        if not terms:
            return "0"
        
        return " + ".join(terms)
class FastPhinaryNumber:
    """
    Fast implementation of base-φ (phinary) numbers using sparse representation
    and efficient normalization algorithms.
    """
    
    PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ≈ 1.618
    PHI_INV = 2 / (1 + math.sqrt(5))  # 1/φ ≈ 0.618
    
    def __init__(self, powers: Union[Set[int], List[int], str, float] = None):
        """
        Initialize using sparse representation - only store the powers that have coefficient 1.
        
        Args:
            powers: Can be:
                - Set/List of integers representing powers of φ with coefficient 1
                - String like "101" (gets converted to powers)
                - Float/int to convert from decimal
        """
        if powers is None:
            self.powers = set()
        elif isinstance(powers, (set, list, tuple)):
            self.powers = set(powers) if not isinstance(powers, set) else powers
        elif isinstance(powers, str):
            # Convert "101" to powers {0, 2}
            self.powers = {len(powers) - 1 - i for i, d in enumerate(powers) if d == '1'}
        elif isinstance(powers, (int, float)):
            self.powers = self._from_decimal_fast(float(powers))
        else:
            raise ValueError("Invalid input type")
        
        self._normalize_fast()
    
    def _from_decimal_fast(self, value: float) -> Set[int]:
        """Fast decimal to phinary conversion using greedy algorithm."""
        if value <= 1e-10:
            return set()
        
        powers = set()
        remaining = value
        
        # Start from a reasonable upper bound
        max_power = int(math.log(value * 2.236) / math.log(self.PHI)) + 1
        
        # Greedy algorithm: use largest possible powers first
        for k in range(max_power, -20, -1):
            phi_k = self.PHI ** k
            if remaining >= phi_k - 1e-10:
                powers.add(k)
                remaining -= phi_k
                if remaining < 1e-10:
                    break
        
        return powers
    
    def _normalize_fast(self):
        """
        Fast normalization using the key insight that consecutive powers
        can be combined in one pass using φ^k + φ^(k+1) = φ^(k+2).
        """
        if not self.powers:
            return
        
        # Convert to sorted list for processing
        sorted_powers = sorted(self.powers, reverse=True)
        new_powers = set()
        
        i = 0
        while i < len(sorted_powers):
            current_power = sorted_powers[i]
            
            # Check if next power is consecutive (current - 1)
            if (i + 1 < len(sorted_powers) and 
                sorted_powers[i + 1] == current_power - 1):
                
                # Combine consecutive powers: φ^(k-1) + φ^k = φ^(k+1)
                combined_power = current_power + 1
                
                # Add the combined power back to our list in the right position
                # This is more efficient than restarting the whole process
                j = 0
                while j < len(sorted_powers) and sorted_powers[j] > combined_power:
                    j += 1
                sorted_powers.insert(j, combined_power)
                
                # Skip the two powers we just combined
                i += 2
            else:
                # Keep this power as is
                new_powers.add(current_power)
                i += 1
        
        self.powers = new_powers
    
    def __add__(self, other: 'FastPhinaryNumber') -> 'FastPhinaryNumber':
        """Fast addition using set operations."""
        if not isinstance(other, FastPhinaryNumber):
            other = FastPhinaryNumber(other)
        
        # Simply combine the power sets - normalization handles overlaps
        combined_powers = self.powers | other.powers
        
        # Handle overlapping powers (where both numbers have the same power)
        overlaps = self.powers & other.powers
        for power in overlaps:
            # Remove the overlap and add the next power (since φ^k + φ^k = 2φ^k)
            combined_powers.remove(power)
            # 2φ^k needs to be handled by normalization, so add it twice conceptually
            # But we can be smarter: 2φ^k = φ^k + φ^k, and φ^k + φ^(k+1) = φ^(k+2)
            # So 2φ^k = φ^(k+2) - φ^(k-1) if k > 0, or just use the relationship
            self._add_power_with_multiplicity(combined_powers, power, 2)
        
        result = FastPhinaryNumber()
        result.powers = combined_powers
        result._normalize_fast()
        return result
    
    def _add_power_with_multiplicity(self, power_set: Set[int], power: int, multiplicity: int):
        """Helper to add a power multiple times, handling the carries efficiently."""
        while multiplicity > 0:
            if multiplicity == 1:
                power_set.add(power)
                break
            elif power in power_set:
                # φ^k + φ^k = 2φ^k, remove current and add to higher power
                power_set.remove(power)
                multiplicity += 1
                power += 1
            else:
                if multiplicity >= 2:
                    # 2φ^k = φ^(k+2) - φ^(k-1) for k >= 1
                    # For k = 0: 2φ^0 = 2 = φ^2
                    # For k = 1: 2φ^1 = 2φ = φ^3 - φ^0
                    if power == 0:
                        power_set.add(2)  # 2 = φ^2 - φ^(-1), but simpler: φ^2
                        multiplicity -= 2
                    elif power == 1:
                        power_set.add(3)  # 2φ = φ^3 - 1, but let's use φ^3
                        if 0 not in power_set:
                            power_set.add(0)
                        else:
                            power_set.remove(0)
                            self._add_power_with_multiplicity(power_set, 1, 1)
                        multiplicity -= 2
                    else:
                        # General case: 2φ^k = φ^(k+2) - φ^(k-1)
                        power_set.add(power + 2)
                        if power - 1 not in power_set:
                            # We need to subtract φ^(k-1), which means we have -φ^(k-1)
                            # This is getting complex, let's use simpler approach
                            pass
                        multiplicity -= 2
                else:
                    power_set.add(power)
                    multiplicity -= 1
    
    def __add__(self, other: 'FastPhinaryNumber') -> 'FastPhinaryNumber':
        """Simplified fast addition."""
        if not isinstance(other, FastPhinaryNumber):
            other = FastPhinaryNumber(other)
        
        # Use a counter approach for cleaner handling
        power_counts = {}
        
        # Count occurrences of each power
        for power in self.powers:
            power_counts[power] = power_counts.get(power, 0) + 1
        
        for power in other.powers:
            power_counts[power] = power_counts.get(power, 0) + 1
        
        # Convert counts back to powers, handling multiples
        result_powers = set()
        for power, count in power_counts.items():
            self._handle_power_count(result_powers, power, count)
        
        result = FastPhinaryNumber()
        result.powers = result_powers
        result._normalize_fast()
        return result
    
    def _handle_power_count(self, power_set: Set[int], power: int, count: int):
        """Handle multiple occurrences of the same power."""
        if count == 1:
            power_set.add(power)
        elif count == 2:
            # 2φ^k = φ^(k+1) + φ^(k-1) when k >= 1
            # For k = 0: 2 = φ^1 + φ^(-2), but let's use Fibonacci relation
            # Actually, let's use: 2φ^k gets converted during normalization
            power_set.add(power)
            power_set.add(power)  # Add twice, let normalize handle it
        else:
            # For higher counts, break down recursively
            power_set.add(power)
            self._handle_power_count(power_set, power, count - 1)
    
    def to_decimal(self) -> float:
        """Fast decimal conversion."""
        return sum(self.PHI ** power for power in self.powers)
    
    def to_binary_string(self) -> str:
        """Convert to binary-like string representation."""
        if not self.powers:
            return "0"
        
        min_power = min(self.powers)
        max_power = max(self.powers)
        
        result = []
        for power in range(max_power, min_power - 1, -1):
            result.append('1' if power in self.powers else '0')
        
        return ''.join(result)
    
    def __str__(self) -> str:
        return self.to_binary_string() + 'φ'
    
    def __repr__(self) -> str:
        return f"FastPhinaryNumber({sorted(self.powers, reverse=True)}[≈{self.to_decimal():.6f}])"
    
    def detailed_repr(self) -> str:
        """Show the detailed mathematical representation."""
        if not self.powers:
            return "0"
        
        terms = []
        for power in sorted(self.powers, reverse=True):
            if power == 0:
                terms.append("1")
            elif power == 1:
                terms.append("φ")
            else:
                terms.append(f"φ^{power}")
        
        return " + ".join(terms)


# Even faster version using bit manipulation concepts
class UltraFastPhinaryNumber:
    """
    Ultra-optimized base-φ (phinary) number system designed for all four arithmetic operations.
    Uses advanced bit manipulation and mathematical properties of the golden ratio.
    """
    
    # Precomputed constants for maximum speed
    PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.618033988749
    PHI_SQUARED = PHI * PHI       # φ² = φ + 1
    PHI_INV = 1 / PHI            # 1/φ = φ - 1 ≈ 0.618033988749
    PHI_INV_SQUARED = PHI_INV * PHI_INV
    
    # Precomputed powers for ultra-fast conversion (up to φ^63)
    _PHI_POWERS = None
    _PHI_INV_POWERS = None
    _MAX_PRECOMPUTED = 64
    
    @classmethod
    def _init_powers(cls):
        """Initialize precomputed power tables."""
        if cls._PHI_POWERS is None:
            cls._PHI_POWERS = [cls.PHI ** i for i in range(cls._MAX_PRECOMPUTED)]
            cls._PHI_INV_POWERS = [cls.PHI_INV ** i for i in range(cls._MAX_PRECOMPUTED)]
    
    def __init__(self, value: Union[int, float, str, 'UltraFastPhinaryNumber'] = 0):
        """Initialize with various input types, optimized for speed."""
        self._init_powers()
        
        if isinstance(value, UltraFastPhinaryNumber):
            self.bits = value.bits
        elif isinstance(value, str):
            self.bits = self._parse_string_fast(value)
        elif isinstance(value, int):
            if value == 0:
                self.bits = 0
            else:
                self.bits = self._from_decimal_optimized(float(value))
        elif isinstance(value, float):
            self.bits = self._from_decimal_optimized(value)
        else:
            self.bits = 0
        
        self._normalize_optimized()
    
    def _parse_string_fast(self, s: str) -> int:
        """Optimized string parsing with validation."""
        if not s or s == '0':
            return 0
        
        # Remove any φ suffix and validate
        s = s.rstrip('φ')
        if not all(c in '01' for c in s):
            raise ValueError(f"Invalid phinary string: {s}")
        
        return int(s, 2) if s else 0
    
    def _from_decimal_optimized(self, value: float) -> int:
        """Ultra-optimized decimal to phinary conversion using precomputed powers."""
        if abs(value) < 1e-15:
            return 0
        
        bits = 0
        remaining = abs(value)
        
        # Use precomputed powers for maximum speed
        for i in range(self._MAX_PRECOMPUTED - 1, -1, -1):
            if remaining >= self._PHI_POWERS[i] - 1e-12:
                bits |= (1 << i)
                remaining -= self._PHI_POWERS[i]
                if remaining < 1e-12:
                    break
        
        return bits
    
    def _normalize_optimized(self):
        """
        Ultra-optimized normalization using advanced bit manipulation.
        Key insight: φ^n + φ^(n-1) = φ^(n+1), so consecutive 1s can be combined.
        """
        if self.bits == 0:
            return
        
        # Single-pass normalization using bit manipulation tricks
        while True:
            # Find all consecutive 1-bit pairs: 11 pattern
            consecutive = self.bits & (self.bits >> 1)
            
            if consecutive == 0:
                break
            
            # Clear the consecutive pairs and add the combined result
            # This is much faster than the previous approach
            self.bits ^= consecutive | (consecutive << 1)  # Clear both positions
            self.bits |= (consecutive << 2)  # Add combined result
    
    def __add__(self, other: Union['UltraFastPhinaryNumber', int, float]) -> 'UltraFastPhinaryNumber':
        """Ultra-optimized addition with support for all numeric types."""
        if not isinstance(other, UltraFastPhinaryNumber):
            other = UltraFastPhinaryNumber(other)
        
        result = UltraFastPhinaryNumber.__new__(UltraFastPhinaryNumber)
        result._init_powers()
        
        # Core addition algorithm optimized for speed
        a, b = self.bits, other.bits
        result.bits = a ^ b  # XOR gives sum without carries
        
        # Handle carries (overlapping bits) efficiently
        carry = (a & b) << 1  # Overlapping bits become carries
        
        # Propagate carries using fast bit manipulation
        while carry:
            temp = result.bits ^ carry
            carry = (result.bits & carry) << 1
            result.bits = temp
        
        result._normalize_optimized()
        return result
    
    def __sub__(self, other: Union['UltraFastPhinaryNumber', int, float]) -> 'UltraFastPhinaryNumber':
        """
        Optimized subtraction using the identity φ^(n+1) = φ^n + φ^(n-1).
        This allows us to "borrow" efficiently in phinary arithmetic.
        """
        if not isinstance(other, UltraFastPhinaryNumber):
            other = UltraFastPhinaryNumber(other)
        
        if self.bits == other.bits:
            return UltraFastPhinaryNumber(0)
        
        result = UltraFastPhinaryNumber.__new__(UltraFastPhinaryNumber)
        result._init_powers()
        
        # Subtraction algorithm using complement and normalization
        minuend = self.bits
        subtrahend = other.bits
        
        # Handle borrowing by expanding higher powers when needed
        while subtrahend and (subtrahend & ~minuend):  # While we need to borrow
            # Find positions where we need to borrow
            need_borrow = subtrahend & ~minuend
            
            # Find the lowest position where we can borrow from
            borrow_from = minuend & -(~minuend | (need_borrow - 1))
            if borrow_from == 0:
                # Need to expand a higher power
                highest_bit = 1 << (minuend.bit_length())
                if highest_bit.bit_length() >= self._MAX_PRECOMPUTED:
                    break  # Prevent overflow
                minuend |= highest_bit
                continue
            
            # Perform the borrow: φ^(n+1) → φ^n + φ^(n-1)
            borrow_pos = (borrow_from & -borrow_from).bit_length() - 1
            minuend &= ~(1 << borrow_pos)  # Remove the borrowed bit
            minuend |= (1 << (borrow_pos - 1)) | (1 << (borrow_pos - 2))  # Add φ^(n-1) + φ^(n-2)
        
        result.bits = minuend ^ subtrahend  # XOR to get the difference
        result._normalize_optimized()
        return result
    
    def __mul__(self, other: Union['UltraFastPhinaryNumber', int, float]) -> 'UltraFastPhinaryNumber':
        """
        Optimized multiplication using the distributive property and bit shifting.
        φ^a * φ^b = φ^(a+b), so we can use convolution-like approach.
        """
        if not isinstance(other, UltraFastPhinaryNumber):
            other = UltraFastPhinaryNumber(other)
        
        if self.bits == 0 or other.bits == 0:
            return UltraFastPhinaryNumber(0)
        
        result = UltraFastPhinaryNumber.__new__(UltraFastPhinaryNumber)
        result._init_powers()
        result.bits = 0
        
        # Multiplication by convolution: each bit in self multiplied by each bit in other
        a_bits = self.bits
        b_bits = other.bits
        
        a_pos = 0
        while a_bits:
            if a_bits & 1:
                # Multiply φ^a_pos by all terms in other
                b_temp = b_bits
                b_pos = 0
                while b_temp:
                    if b_temp & 1:
                        # φ^a_pos * φ^b_pos = φ^(a_pos + b_pos)
                        power_sum = a_pos + b_pos
                        if power_sum < 64:  # Prevent overflow
                            result.bits |= (1 << power_sum)
                    b_temp >>= 1
                    b_pos += 1
            a_bits >>= 1
            a_pos += 1
        
        result._normalize_optimized()
        return result
    
    def __truediv__(self, other: Union['UltraFastPhinaryNumber', int, float]) -> 'UltraFastPhinaryNumber':
        """
        Optimized division using the property φ^a / φ^b = φ^(a-b).
        This is implemented as repeated subtraction with optimizations.
        """
        if not isinstance(other, UltraFastPhinaryNumber):
            other = UltraFastPhinaryNumber(other)
        
        if other.bits == 0:
            raise ZeroDivisionError("Division by zero in phinary arithmetic")
        
        if self.bits == 0:
            return UltraFastPhinaryNumber(0)
        
        # For now, convert to decimal, divide, and convert back
        # This can be optimized further with direct phinary long division
        decimal_result = self.to_decimal() / other.to_decimal()
        return UltraFastPhinaryNumber(decimal_result)
    
    def __pow__(self, exponent: Union[int, 'UltraFastPhinaryNumber']) -> 'UltraFastPhinaryNumber':
        """Fast exponentiation using binary exponentiation algorithm."""
        if isinstance(exponent, UltraFastPhinaryNumber):
            exp = int(exponent.to_decimal())
        else:
            exp = int(exponent)
        
        if exp == 0:
            return UltraFastPhinaryNumber(1)
        if exp == 1:
            return UltraFastPhinaryNumber(self)
        
        result = UltraFastPhinaryNumber(1)
        base = UltraFastPhinaryNumber(self)
        
        while exp > 0:
            if exp & 1:
                result = result * base
            base = base * base
            exp >>= 1
        
        return result
    
    def to_decimal(self) -> float:
        """Ultra-fast decimal conversion using precomputed powers."""
        if self.bits == 0:
            return 0.0
        
        result = 0.0
        bits = self.bits
        pos = 0
        
        # Use precomputed powers for maximum speed
        while bits and pos < self._MAX_PRECOMPUTED:
            if bits & 1:
                result += self._PHI_POWERS[pos]
            bits >>= 1
            pos += 1
        
        return result
    
    def to_string(self, reverse: bool = True) -> str:
        """Fast string conversion with optional bit order."""
        if self.bits == 0:
            return "0"
        
        binary_str = bin(self.bits)[2:]
        return binary_str[::-1] if reverse else binary_str
    
    def __str__(self) -> str:
        return self.to_string() + 'φ'
    
    def __repr__(self) -> str:
        return f"UltraFast({self.to_string()}φ≈{self.to_decimal():.6f})"
    
    def __eq__(self, other: Union['UltraFastPhinaryNumber', int, float]) -> bool:
        if isinstance(other, UltraFastPhinaryNumber):
            return self.bits == other.bits
        return abs(self.to_decimal() - float(other)) < 1e-10
    
    def __lt__(self, other: Union['UltraFastPhinaryNumber', int, float]) -> bool:
        if isinstance(other, UltraFastPhinaryNumber):
            return self.to_decimal() < other.to_decimal()
        return self.to_decimal() < float(other)
    
    def __hash__(self) -> int:
        return hash(self.bits)
    
    # Optimized utility methods
    def is_zero(self) -> bool:
        """Fast zero check."""
        return self.bits == 0
    
    def bit_length(self) -> int:
        """Return the number of bits required to represent this number."""
        return self.bits.bit_length()
    
    def copy(self) -> 'UltraFastPhinaryNumber':
        """Fast copy operation."""
        new_num = UltraFastPhinaryNumber.__new__(UltraFastPhinaryNumber)
        new_num._init_powers()
        new_num.bits = self.bits
        return new_num
    
    # Advanced operations for complex calculations
    def shift_left(self, positions: int) -> 'UltraFastPhinaryNumber':
        """Multiply by φ^positions (shift left in phinary)."""
        result = self.copy()
        result.bits <<= positions
        result._normalize_optimized()
        return result
    
    def shift_right(self, positions: int) -> 'UltraFastPhinaryNumber':
        """Divide by φ^positions (shift right in phinary)."""
        result = self.copy()
        result.bits >>= positions
        return result


# Example usage and testing
if __name__ == "__main__":
    print("=== Base-φ (Phinary) Number System ===\n")
    
    # Create some numbers
    print("Creating phinary numbers:")
    a = PhinaryNumber("1")      # 1 = φ^0
    b = PhinaryNumber("10")     # φ^1 = φ
    c = PhinaryNumber("100")    # φ^2
    d = PhinaryNumber("1001")   # φ^3 + φ^0
    
    numbers = [a, b, c, d]
    for num in numbers:
        print(f"{num} = {num.detailed_repr()}")
    
    print(f"\nNote: φ ≈ {PhinaryNumber.PHI:.6f}")
    
    print("\n=== Addition Examples ===")
    
    # Example 1: 1 + φ
    result1 = a + b
    print(f"\n1 + φ:")
    print(f"{a} + {b} = {result1}")
    print(f"Detailed: {a.detailed_repr()} + {b.detailed_repr()} = {result1.detailed_repr()}")
    
    # Example 2: φ + φ (should demonstrate normalization)
    result2 = b + b
    print(f"\nφ + φ:")
    print(f"{b} + {b} = {result2}")
    print(f"Detailed: {b.detailed_repr()} + {b.detailed_repr()} = {result2.detailed_repr()}")
    
    # Example 3: More complex addition
    result3 = c + d
    print(f"\nφ² + (φ³ + 1):")
    print(f"{c} + {d} = {result3}")
    print(f"Detailed: {c.detailed_repr()} + {d.detailed_repr()} = {result3.detailed_repr()}")

    long_number1 = PhinaryNumber("10101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010101001000101001010101011010101010010010100100000101010000101000100101001001010100000100101010010100101001001001101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101")
    long_number2 = PhinaryNumber("10101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010101001000101001010101011010101010010010100100000101010000101000100101001001010100000100101010010100101001001001101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010101001000101001010101011010101010010010100100000101010000101000100101001001010100000100101010010100101001001001101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010101001000101001010101011010101010010010100100000101010000101000100101001001010100000100101010010100101001001001101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101")
    a = 10101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010101001000101001010101011010101010010010100100000101010000101000100101001001010100000100101010010100101001001001101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010101001000101001010101011010101010010010100100000101010000101000100101001001010100000100101010010100101001001001101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010101001000101001010101011010101010010010100100000101010000101000100101001001010100000100101010010100101001001001101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101
    b = 10101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010101001000101001010101011010101010010010100100000101010000101000100101001001010100000100101010010100101001001001101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101
    fast_long_number1 = FastPhinaryNumber("10101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010101001000101001010101011010101010010010100100000101010000101000100101001001010100000100101010010100101001001001101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101")
    fast_long_number2 = FastPhinaryNumber("10101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010101001000101001010101011010101010010010100100000101010000101000100101001001010100000100101010010100101001001001101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010101001000101001010101011010101010010010100100000101010000101000100101001001010100000100101010010100101001001001101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010101001000101001010101011010101010010010100100000101010000101000100101001001010100000100101010010100101001001001101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101")
    ultra_fast_long_number1 = UltraFastPhinaryNumber("10101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010101001000101001010101011010101010010010100100000101010000101000100101001001010100000100101010010100101001001001101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101")
    ultra_fast_long_number2 = UltraFastPhinaryNumber("10101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010101001000101001010101011010101010010010100100000101010000101000100101001001010100000100101010010100101001001001101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010101001000101001010101011010101010010010100100000101010000101000100101001001010100000100101010010100101001001001101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101101010101010100100010100101010101101010101001001010010000010101000010100010010100100101010000010010101001010010100100100110101010101001001010101001010100101010100101010101011110100110010101010101011010100011001010110100101010101000101010101011010101010101001000101001010101011010101010010010100100000101010000101000100101001001010100000100101010010100101001001001101010101010010010101010010101001010101001010101010111101001100101010101010110101000110010101101001010101010001010101010110101010101010010001010010101010110101010100100101001000001010100001010001001010010010101000001001010100101001010010010011010101010100100101010100101010010101010010101010101111010011001010101010101101010001100101011010010101010100010101010101")
    
    print("Phinary additon")
    # start = time.time()
    # for _ in range(1000):
    #     result = long_number1 + long_number2
    # print(f"Normal Phinary Time taken: {time.time() - start}")
    start = time.time()
    for _ in range(1000):
        result2 = a + b
    print(f"Normal int Time taken: {time.time() - start}")
    # start = time.time()
    # for _ in range(1000):
    #     result = fast_long_number1 + fast_long_number2
    # print(f"Fast Phinary Time taken: {time.time() - start}")
    start = time.time()
    for _ in range(10000):
        result2 = ultra_fast_long_number1 + ultra_fast_long_number2
    print(f"Ultra fast Phinary Time taken: {time.time() - start}")
