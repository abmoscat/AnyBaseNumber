import math

class Phinary:
    """
    Represents a number in base phi (the golden ratio).

    A number X is represented as a sum of distinct powers of phi:
    X = sum_{k} d_k * phi^k
    where d_k can only be {0, 1}.
    The representation can have a 'phinary point', similar to a binary
    or decimal point. Digits to the left of the point correspond to
    non-negative powers of phi (phi^0, phi^1, phi^2, ...), and digits
    to the right correspond to negative powers of phi (phi^-1, phi^-2, ...).

    Standard form: In base phi, some numbers have multiple representations.
    A common standard form (non-adjacent form or NAF) ensures that no two
    consecutive '1's appear in the representation (i.e., '11' is replaced
    by '100' since phi^n + phi^(n-1) = phi^(n+1)). This class will aim
    to store and work with this standard form where appropriate, although
    initial conversion from a float might not immediately produce it without
    a normalization step.
    """

    PHI = (1 + math.sqrt(5)) / 2

    def __init__(self, value, precision=30):
        """
        Initializes a Phinary number.

        Args:
            value: Can be:
                - A string representation in phinary (e.g., "10.101").
                - A standard Python int or float to be converted to phinary.
                - A list or tuple of integers (0s and 1s) representing digits,
                  and an integer indicating the position of the phinary point.
                  e.g., ([1,0,0,1], 2) for "10.01" (point after 2nd digit from left)
            precision (int): The number of digits to compute after the phinary
                             point when converting from a float. Also used as a
                             limit for representation length in some cases.
        """
        self.digits = []  # List of integers (0 or 1)
        self.point_position = 0 # Index of the digit before the phinary point.
                                # e.g., for 101.01, digits=[1,0,1,0,1], point_position=3

        if isinstance(value, str):
            self._from_string(value)
        elif isinstance(value, (int, float)):
            self._from_float(value, precision)
        elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], list) and isinstance(value[1], int):
            # Potentially add more validation for digits being 0 or 1
            self.digits = value[0][:]
            self.point_position = value[1]
            # self._normalize() # Optional: normalize upon creation
        else:
            raise TypeError("Unsupported type for Phinary initialization. "
                            "Use string, int/float, or (list_of_digits, point_pos).")

    def _from_string(self, s_value: str):
        """Converts a string representation (e.g., "10.101") to internal digits."""
        if not all(c in '01.' for c in s_value):
            raise ValueError("Phinary string can only contain '0', '1', and at most one '.'")
        if s_value.count('.') > 1:
            raise ValueError("Phinary string can have at most one '.'")

        if '.' in s_value:
            integer_part_str, fractional_part_str = s_value.split('.')
            self.point_position = len(integer_part_str) if integer_part_str else 0
            all_digits_str = integer_part_str + fractional_part_str
        else:
            integer_part_str = s_value
            fractional_part_str = ""
            self.point_position = len(integer_part_str)
            all_digits_str = integer_part_str

        if not all_digits_str: # Handles cases like "." or ""
             raise ValueError("Phinary string must contain digits.")

        self.digits = [int(d) for d in all_digits_str]
        # self._normalize() # Optional

    def _from_float(self, float_value: float, precision: int):
        """
        Converts a standard Python float to its phinary representation.
        This is a greedy algorithm and might not always produce the shortest
        or standard (non-adjacent) form without a subsequent normalization step.
        """
        if float_value < 0:
            # This representation is typically for positive numbers.
            # Handling negative numbers would require an additional sign attribute.
            raise ValueError("Phinary representation is typically for positive numbers.")

        # Determine the largest power of PHI less than or equal to float_value
        if float_value == 0:
            self.digits = [0]
            self.point_position = 1
            return

        max_power = 0
        if float_value >= 1:
            while Phinary.PHI ** max_power <= float_value:
                max_power += 1
            max_power -=1 # Went one too far
        else: # 0 < float_value < 1
            max_power = -1
            while Phinary.PHI ** max_power > float_value:
                max_power -=1

        self.digits = []
        current_power = max_power
        self.point_position = max(0, max_power + 1)
        num_digits_generated = 0

        temp_value = float_value

        # Iterate for integer part and fractional part up to precision
        # The loop condition needs to ensure we generate enough digits.
        # For the fractional part, we go down to max_power - precision + (max_power + 1 if max_power >=0 else 0)
        # Essentially, we want 'precision' digits after the point.
        # The point is at index max_power+1 if max_power >=0, or 0 if max_power < 0.

        # Example: float_value = 7, PHI ~ 1.618
        # PHI^0=1, PHI^1=1.618, PHI^2=2.618, PHI^3=4.236, PHI^4=6.854, PHI^5=11.09
        # max_power = 4
        # point_position = 5 (digits for powers 4, 3, 2, 1, 0 before point)

        # Example: float_value = 0.5
        # PHI^-1=0.618, PHI^-2=0.382
        # max_power = -1
        # point_position = 0 (all digits after point)

        # We will generate digits from power `max_power` down to `max_power - total_digits_to_generate + 1`
        # `total_digits_to_generate` needs to cover from `max_power` down to some negative power
        # dictated by `precision` for the fractional part.

        # Let's define the lowest power we consider:
        # If max_power >= 0, point is after digits representing powers down to 0.
        #   Fractional digits start from power -1. We need `precision` of these.
        #   So lowest_power = -precision
        # If max_power < 0, all digits are fractional. Point is at the beginning.
        #   The digit for `max_power` is the first fractional digit.
        #   We need `precision` digits in total, starting from `max_power`.
        #   So lowest_power = max_power - precision + 1.
        # More simply: the highest power is `max_power`. The lowest power we care about
        # for the fractional part is roughly `-precision`.
        # Total number of powers to check: max_power - (-precision) + 1 = max_power + precision + 1

        lowest_power_to_check = min(-1, max_power) - precision # Ensure enough fractional digits
        if max_power >=0 :
            lowest_power_to_check = -precision # We want precision digits after the point (phi^0)

        # Adjust point_position if all digits are fractional initially
        if max_power < 0:
            self.digits.extend([0] * abs(max_power + 1)) # Add leading zeros for powers like -1, -2...
            self.point_position = 0 # All fractional

        # Iterate from the highest relevant power down to a suitable lowest power
        # This part needs refinement to correctly align digits and point_position.

        current_val = float_value
        # Integer part (and potentially start of fractional if float_value < 1)
        p = max_power
        while p >= 0 and num_digits_generated < (max_power + 1 + precision): # Iterate through powers >= 0
            phi_p = Phinary.PHI ** p
            if current_val >= phi_p - 1e-9: # Using tolerance for float comparison
                self.digits.append(1)
                current_val -= phi_p
            else:
                self.digits.append(0)
            p -= 1
            num_digits_generated +=1

        if not self.digits and max_power < 0: # e.g. converting 0.5, max_power = -1
             # point_position is 0. digits should start for phi^-1, phi^-2 etc.
             pass # Handled by fractional part loop

        # Fractional part
        # If point_position was set based on positive max_power, it's correct.
        # If max_power < 0, point_position is 0.
        # We need to ensure self.digits has placeholders up to point_position if it's > 0
        # and point_position wasn't naturally filled by positive powers.

        # Correctly establish point_position based on max_power before appending fractional
        if max_power >=0:
            self.point_position = max_power + 1
        else: # max_power < 0
            self.point_position = 0
            # Add leading zeros for powers like phi^-1, phi^-2 if max_power was, say, -3
            # The first digit corresponds to phi^(max_power)
            # self.digits should be empty here if we only had fractional part
            # We need to fill up to `abs(max_power+1)` leading zeros in `digits` if `point_position` is 0
            # and `max_power` is e.g. -2 (so we need a 0 for phi^-1 then digit for phi^-2)
            # This logic is tricky. Let's simplify:
            # Construct the full list of digits first, then set point_position.

        # Reset and rebuild based on a simpler greedy approach:
        self.digits = []
        temp_value = float_value

        # Determine initial power (can be negative)
        current_p = 0
        if temp_value == 0:
            self.digits = [0]
            self.point_position = 1
            return

        if temp_value >= 1:
            while Phinary.PHI ** current_p <= temp_value:
                current_p += 1
            current_p -= 1
        else: # 0 < temp_value < 1
            current_p = -1
            while Phinary.PHI ** current_p > temp_value: # Find first power that is smaller
                current_p -= 1

        self.point_position = max(0, current_p + 1) # Position after highest power digit

        # Generate digits from current_p down to a limit ensuring precision
        # Limit for powers: current_p - (point_position related digits) - precision
        # Iterate for a fixed number of fractional digits
        # max_power_idx = current_p # save for later
        
        # Digits for powers >= 0
        p_iter = current_p
        while p_iter >= 0:
            phi_val = Phinary.PHI ** p_iter
            if temp_value >= phi_val - 1e-9: # Tolerance
                self.digits.append(1)
                temp_value -= phi_val
            else:
                self.digits.append(0)
            p_iter -= 1

        # If current_p was < 0, self.digits is still empty, point_position is 0.
        # Add leading zeros for the fractional part if needed, e.g. 0.01 (phi^-2)
        if current_p < 0:
            # For current_p = -1, first digit is for phi^-1
            # For current_p = -2, first digit is for phi^-2, need a 0 for phi^-1
            self.digits.extend([0] * abs(current_p + 1))


        # Digits for powers < 0 (fractional part)
        for i in range(precision):
            p_iter = -(i + 1) # phi^-1, phi^-2, ...
            # Only add digit if we haven't already covered this power due to a negative current_p
            if current_p >=0 or p_iter < current_p : # if current_p was -1, p_iter starts at -1, so only add for -2, -3...
                                                    # if current_p was -3, p_iter starts at -1, we'd fill 0 for -1, 0 for -2, then from -3
                phi_val = Phinary.PHI ** p_iter
                if temp_value >= phi_val - 1e-9: # Tolerance
                    self.digits.append(1)
                    temp_value -= phi_val
                else:
                    self.digits.append(0)
            elif p_iter >= current_p and current_p < 0 and len(self.digits) < abs(current_p+1) + i +1 : # Case where current_p was negative
                # The leading fractional zeros up to current_p are handled.
                # Now append for powers below current_p.
                idx_in_digits = abs(current_p+1) + i # expected position of current p_iter digit
                # Check if this position already determined
                # This section is becoming very complex, the greedy approach needs careful state management.

        # --- A slightly more robust greedy approach for _from_float ---
        self.digits = []
        if float_value == 0:
            self.digits = [0]
            self.point_position = 1 # Represents "0."
            return

        # Find the largest power of PHI <= float_value
        p = 0
        if float_value >= 1:
            while Phinary.PHI ** p <= float_value:
                p += 1
            p -= 1
        else: # 0 < float_value < 1
            p = -1
            # Find the first power of PHI that is <= float_value
            # e.g. 0.5: PHI^-1 = 0.618 (too big), PHI^-2 = 0.382 (ok) -> p = -2
            while Phinary.PHI ** p > float_value and p > -(precision + 5): # Add limit to prevent infinite loop for tiny numbers
                p -= 1

        self.point_position = max(0, p + 1) # e.g. p=2 (phi^2), pp=3. p=-1 (phi^-1), pp=0.

        current_val = float_value
        num_digits_generated = 0
        max_digits = (p + 1 if p >=0 else 0) + precision # total digits: integer part + fractional part

        # Iterate from power p down to a suitable negative power
        # The digits list will store coefficients from d_p, d_{p-1}, ..., d_0, d_{-1}, ...
        power_iterator = p
        while num_digits_generated < max_digits :
            # If p becomes too small and current_val is near zero, break
            if power_iterator < -(precision + 5) and abs(current_val) < 1e-12: # Heuristic break
                break

            phi_power_val = Phinary.PHI ** power_iterator
            if current_val >= phi_power_val - 1e-9: # Tolerance
                self.digits.append(1)
                current_val -= phi_power_val
            else:
                self.digits.append(0)

            power_iterator -= 1
            num_digits_generated +=1

        # Trim trailing zeros from fractional part if they don't change value
        # Only if there is a fractional part
        if self.point_position < len(self.digits): # implies fractional part exists
            # Find last '1' in fractional part
            last_one_in_fractional = -1
            for i in range(len(self.digits) - 1, self.point_position - 1, -1):
                if self.digits[i] == 1:
                    last_one_in_fractional = i
                    break
            if last_one_in_fractional != -1:
                self.digits = self.digits[:last_one_in_fractional + 1]
            else: # All zeros in fractional part
                self.digits = self.digits[:self.point_position]


        # Ensure there's at least a '0' if digits became empty (e.g. for very small numbers rounding to 0)
        if not self.digits:
            self.digits = [0]
            self.point_position = 1 # Represents "0." or just "0" if pp was 1
            if p < 0 : self.point_position = 0 # e.g. for value like 1e-20 that became [0]

        # Ensure point_position is not out of bounds if all digits are fractional and leading
        # Example: 0.1 (phinary) from a float. p=-1. pp=0. digits=[1]. Correct.
        # Example: 0.01 (phinary) from a float. p=-2. pp=0. digits=[0,1]. Correct. (d_{-1}=0, d_{-2}=1)

        # self._normalize() # Crucial for standard form

    def _normalize(self):
        """
        Converts the internal digit representation to a standard form,
        typically the non-adjacent form (NAF), where "11" is replaced by "100".
        This process might need to ripple through the number and can change
        the number of digits and the point position.
        This is a complex algorithm. For now, we'll placeholder it.
        A common rule: 11_phi = 100_phi (because phi^n + phi^(n-1) = phi^(n+1))
        And also 0...010...0 where the 1 is phi^k, can sometimes be expanded using
        phi^k = phi^(k-1) + phi^(k-2).
        The goal is usually to eliminate "11" patterns.
        """
        # This is a simplified normalization focusing on "11" -> "100"
        # It needs to be done carefully, potentially from right to left (LSB).
        # And it can affect the point_position if digits are added/removed at the ends.
        # For example: "0.11" becomes "1.00" (phi^-1 + phi^-2 = phi^0 = 1)
        # This requires careful handling of powers and point_position.

        # For now, this method is a placeholder. A full normalization
        # for base phi is non-trivial to implement correctly and efficiently.
        # print("Warning: Normalization is not fully implemented.")
        pass


    def to_float(self):
        """Converts the phinary number back to a standard Python float."""
        if not self.digits:
            return 0.0

        value = 0.0
        for i, digit in enumerate(self.digits):
            if digit == 1:
                power = self.point_position - 1 - i
                value += Phinary.PHI ** power
            elif digit != 0:
                raise ValueError("Invalid digit in Phinary representation. Only 0 or 1 allowed.")
        return value

    def __str__(self):
        """Returns the string representation of the phinary number (e.g., "10.101")."""
        if not self.digits:
            return "0" # Or perhaps "0.0" if point_position implies fraction

        # Ensure point_position is within reasonable bounds for display
        # Handle cases where digits might be all fractional [0,0,1] with pp=0 => "0.001"
        # Or pp > len(digits) => "10100." (trailing zeros for integer part)

        s = ""
        if self.point_position == 0: # All fractional, e.g. 0.xxxx
            s = "0." + "".join(map(str, self.digits))
        elif self.point_position == len(self.digits): # All integer, e.g., xxxx.
            s = "".join(map(str, self.digits))
            if not s: s = "0" # if digits was empty and pp somehow > 0
            # s += "." # Conventionally, an integer might not show trailing point
        elif self.point_position > len(self.digits): # Integer with trailing implicit zeros
            s = "".join(map(str, self.digits))
            s += "0" * (self.point_position - len(self.digits))
            # s += "."
        else: # Mixed number xxxx.yyyy
            s = "".join(map(str, self.digits[:self.point_position]))
            s += "."
            s += "".join(map(str, self.digits[self.point_position:]))

        # Cleanup: if string is just ".", make it "0."
        # if s == ".": s = "0." (should be handled by logic above)
        # if s.startswith("."): s = "0" + s (handled by pp=0 case)
        # if s.endswith(".") and len(s) > 1: s = s[:-1] # remove trailing point for pure integers for now
        # Let's keep it simple: if fractional part is empty, don't add "." and trailing digits.

        if not self.digits: return "0"

        int_part_str = ""
        frac_part_str = ""

        if self.point_position > 0:
            int_part_str = "".join(map(str, self.digits[:self.point_position]))
        else: # point_position is 0 or negative (not really supported for pp<0)
            int_part_str = "0"

        if self.point_position < len(self.digits):
            frac_part_str = "".join(map(str, self.digits[self.point_position:]))

        if not int_part_str and self.point_position > 0: # e.g. pp=2, digits=[] (should not happen)
            int_part_str = "0" * self.point_position


        if frac_part_str:
            return f"{int_part_str}.{frac_part_str}"
        else:
            return int_part_str if int_part_str else "0"


    def __repr__(self):
        return f"Phinary('{self.__str__()}')"

    # --- Potentially, arithmetic operations ---
    # These are complex due to the base and the normalization requirement.
    # For example, adding two phinary numbers might result in digits > 1,
    # or "11" patterns, which then need to be normalized.

    def __add__(self, other):
        """Adds two Phinary numbers. (Placeholder - Complex implementation)"""
        if not isinstance(other, Phinary):
            # Could try to convert 'other' if it's float/int to Phinary first
            return NotImplemented
        # 1. Convert both to float
        # 2. Add floats
        # 3. Convert result back to Phinary
        # This is simpler but loses precision and phinary properties during calculation.
        # A direct phinary addition is much harder.
        sum_float = self.to_float() + other.to_float()
        return Phinary(sum_float, precision=max(self._get_precision(), other._get_precision()))

    def _get_precision(self):
        """Helper to estimate current precision."""
        if self.point_position < len(self.digits):
            return len(self.digits) - self.point_position
        return 0

    def __eq__(self, other):
        """Compares two Phinary numbers for equality."""
        if not isinstance(other, Phinary):
            # Optionally convert 'other' to Phinary if it's a compatible type
            try:
                other = Phinary(other) # This might be problematic for comparison if precision differs
            except (TypeError, ValueError):
                return NotImplemented

        # For true equality, they should ideally be in a canonical (normalized) form.
        # Without full normalization, comparing the float values is a practical approach,
        # but subject to floating point inaccuracies.
        # A more rigorous comparison would compare normalized digit lists.
        # Let's use a tolerance for float comparison.
        # self.normalize() # ensure self is normalized
        # other.normalize() # ensure other is normalized
        # return self.digits == other.digits and self.point_position == other.point_position
        # The above is ideal but depends on a perfect _normalize().

        # Practical comparison via float conversion with tolerance:
        # This has issues if different phinary representations map to "almost" the same float.
        tolerance = 1e-9 # Adjust as needed
        return abs(self.to_float() - other.to_float()) < tolerance


    # Other methods to consider:
    # __sub__, __mul__, __div__ (all complex)
    # Comparison operators: __lt__, __le__, __gt__, __ge__
    # normalize() method (crucial for canonical form and some operations)
    # is_standard_form() method

if __name__ == '__main__':
    print(f"Phi = {Phinary.PHI}")

    # Test cases
    p1 = Phinary("10.1") # phi^1 + phi^-1 = 1.618... + 0.618... = 2.236...
    print(f"p1 ('10.1'): {p1} -> {p1.to_float()}") # Expected: 1*PHI + 0*1 + 1/PHI = PHI + PHI^-1 = PHI + (PHI-1) = 2*PHI -1 = sqrt(5) approx 2.236

    p_phi = Phinary(Phinary.PHI)
    print(f"p_phi (from float PHI): {p_phi} -> {p_phi.to_float()}") # Should be close to "10.0" or "1.11..." before norm. "10" is standard.

    p_one = Phinary(1.0)
    print(f"p_one (from float 1.0): {p_one} -> {p_one.to_float()}") # Should be "1" or "0.11" (std is "1")

    p_two = Phinary(2.0)
    print(f"p_two (from float 2.0): {p_two} -> {p_two.to_float()}") # Should be "10.01" (phi + phi^-2) or "1.11..." -> "10.00" (phi^1 + phi^0) ... actually 2 = PHI + 1/PHI^2 = 10.01
                                                                   # 2 = phi + (2-phi) = phi + (2-(1+sqrt(5))/2) = phi + ( (4-1-sqrt(5))/2 ) = phi + (3-sqrt(5))/2
                                                                   # (3-sqrt(5))/2 = phi^-2. So 2 = phi^1 + phi^-2 = "10.01"

    p_three = Phinary(3.0) # 3 = phi^2 + phi^-2
    print(f"p_three (from float 3.0): {p_three} -> {p_three.to_float()}") # Expected: "100.01"

    p_four = Phinary(4.0) # 4 = phi^2 + phi^0 + phi^-2
    print(f"p_four (from float 4.0): {p_four} -> {p_four.to_float()}") # Expected: "101.01"

    p_sqrt5 = Phinary(math.sqrt(5)) # sqrt(5) = 2*PHI - 1 = PHI + (PHI-1) = PHI + PHI^-1
    print(f"p_sqrt5 (from float sqrt(5)): {p_sqrt5} -> {p_sqrt5.to_float()}") # Expected: "10.1"

    # Test initialization from string
    p_str = Phinary("100.01") # phi^2 + phi^-2 = 2.618... + 0.382... = 3
    print(f"p_str ('100.01'): {p_str} -> {p_str.to_float()}")

    p_frac = Phinary("0.1") # phi^-1
    print(f"p_frac ('0.1'): {p_frac} -> {p_frac.to_float()}")

    p_frac_zero = Phinary("0.00")
    print(f"p_frac_zero ('0.00'): {p_frac_zero} -> {p_frac_zero.to_float()}")

    p_int_zero = Phinary("0")
    print(f"p_int_zero ('0'): {p_int_zero} -> {p_int_zero.to_float()}")

    # Test conversion of small numbers
    p_small = Phinary(0.5, precision=10)
    # 0.5 = phi^-2 + phi^-4 + ... (approx 0.3819 + 0.145... ) or 0.5 = phi^-1 - phi^-3 ...
    # Greedy:
    # 0.5. Largest power of phi <= 0.5 is phi^-2 = 0.381966...
    # Remainder: 0.5 - 0.381966 = 0.118034
    # Largest power <= 0.118034 is phi^-5 = 0.09017...
    # Remainder: 0.118034 - 0.09017 = 0.02786...
    # Largest power <= 0.02786 is phi^-8 = 0.0206...
    # So, expect something like "0.01001001..." (d_-2=1, d_-3=0, d_-4=0, d_-5=1 ...)
    print(f"p_small (from float 0.5): {p_small} -> {p_small.to_float()}")

    # Test addition (uses float conversion, so approximate)
    p_add1 = Phinary("10.1") # ~2.236 (sqrt(5))
    p_add2 = Phinary("1.0")  # 1.0
    p_sum = p_add1 + p_add2
    print(f"Sum of {p_add1} and {p_add2}: {p_sum} -> {p_sum.to_float()} (expected ~3.236)")
    # sqrt(5)+1 = PHI+PHI^-1+1 = PHI+PHI^-1+PHI^0 = 10.1 + 1 = 11.1 (not standard)
    # 11.1 -> 100.1 (phi + phi^0 + phi^-1 = phi^2 + phi^-1)
    # phi^2+phi^-1 = 2.618 + 0.618 = 3.236

    # Test equality
    p_eq1 = Phinary(2.0)
    p_eq2 = Phinary("10.01") # Standard form of 2
    print(f"Is {p_eq1} == {p_eq2}? {p_eq1 == p_eq2}") # True if float conversion is close enough

    p_eq3 = Phinary(Phinary.PHI)
    # The greedy conversion of PHI might give "1.1111..." or "10.000..." depending on precision and algo.
    # Standard form of PHI is "10."
    # p_eq4_str = "10" # if _from_float gives exactly [1,0] and point_position=2
    # p_eq4 = Phinary(p_eq4_str)
    # print(f"Is {p_eq3} ({p_eq3}) == {p_eq4} ({p_eq4})? {p_eq3 == p_eq4}")
    # This highlights the need for normalization for robust equality.

    # Test number like 0.01_phi = phi^-2
    p_phi_neg2_str = Phinary("0.01")
    print(f"p_phi_neg2_str ('0.01'): {p_phi_neg2_str} -> {p_phi_neg2_str.to_float()} (actual phi^-2: {Phinary.PHI**-2})")

    p_phi_neg2_float = Phinary(Phinary.PHI**-2)
    print(f"p_phi_neg2_float (from float phi^-2): {p_phi_neg2_float} -> {p_phi_neg2_float.to_float()}")

    print(f"Comparing string and float init for phi^-2: {p_phi_neg2_str == p_phi_neg2_float}")