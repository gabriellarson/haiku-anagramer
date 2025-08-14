#!/usr/bin/env python3
"""
Haiku Anagram Generator
Takes a haiku as input, validates it, and attempts to create an anagram 
that is also a valid haiku.
"""

import pronouncing
from collections import Counter
from itertools import combinations, permutations
import re
from typing import List, Optional, Set, Tuple
import nltk
from nltk.corpus import cmudict

# Download required NLTK data if not already present
try:
    cmu_dict = cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    cmu_dict = cmudict.dict()

class HaikuAnagrammer:
    def __init__(self):
        self.cmu_dict = set(cmu_dict.keys())
        # Add common words that might be missing
        self.cmu_dict.update(['a', 'i', 'the', 'and', 'or', 'but'])
        
    def count_syllables(self, word: str) -> int:
        word_lower = word.lower().strip(".,!?;:'\"")
        
        # Try pronouncing package first
        phones_list = pronouncing.phones_for_word(word_lower)
        if phones_list:
            return pronouncing.syllable_count(phones_list[0])
        
        # Fallback: simple vowel counting heuristic
        vowels = 'aeiouAEIOU'
        word_clean = word_lower.strip()
        count = 0
        previous_was_vowel = False
        
        for char in word_clean:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent e
        if word_clean.endswith('e') and count > 1:
            count -= 1
        
        return max(1, count)  # Every word has at least 1 syllable
    
    def validate_haiku(self, text: str) -> Tuple[bool, List[int]]:
        lines = text.strip().split('\n')
        
        if len(lines) != 3:
            return False, []
        
        syllable_counts = []
        expected = [5, 7, 5]
        
        for i, line in enumerate(lines):
            words = line.split()
            line_syllables = sum(self.count_syllables(word) for word in words)
            syllable_counts.append(line_syllables)
        
        is_valid = syllable_counts == expected
        return is_valid, syllable_counts
    
    def get_letter_counts(self, text: str) -> Counter:
        letters_only = re.sub(r'[^a-zA-Z]', '', text.lower())
        return Counter(letters_only)
    
    def is_valid_word(self, word: str) -> bool:
        return word.lower() in self.cmu_dict
    
    def find_words_from_letters(self, letters: Counter, max_words: int = 100000) -> Set[str]:
        valid_words = set()
        letter_string = ''.join(letters.elements())
        
        # Check dictionary words that could potentially be formed
        for word in self.cmu_dict:
            if len(word) <= len(letter_string) and len(word) >= 2:
                word_counter = Counter(word)
                if all(letters[char] >= count for char, count in word_counter.items()):
                    valid_words.add(word)
                    if len(valid_words) >= max_words:
                        break
        
        return valid_words
    
    def generate_haiku_anagrams(self, original_text: str, max_attempts: int = 1000) -> Optional[str]:
        letter_counts = self.get_letter_counts(original_text)
        total_letters = sum(letter_counts.values())
        
        # Find candidate words
        candidate_words = self.find_words_from_letters(letter_counts)
        
        if len(candidate_words) < 3:
            return None  # Not enough words to work with
        
        # Group words by syllable count for efficiency
        words_by_syllables = {}
        for word in candidate_words:
            syll_count = self.count_syllables(word)
            if syll_count not in words_by_syllables:
                words_by_syllables[syll_count] = []
            words_by_syllables[syll_count].append(word)
        
        # Try to build haiku lines (5-7-5 pattern)
        attempts = 0
        
        for _ in range(max_attempts):
            attempts += 1
            
            # Try to build each line
            haiku_lines = []
            remaining_letters = letter_counts.copy()
            
            for target_syllables in [5, 7, 5]:
                line = self.build_line(words_by_syllables, remaining_letters, target_syllables)
                if line is None:
                    break
                haiku_lines.append(line)
                
                # Update remaining letters
                for word in line:
                    for char in word.lower():
                        if char.isalpha():
                            remaining_letters[char] -= 1
            
            # Check if we used all letters and have valid haiku
            if len(haiku_lines) == 3 and all(count == 0 for count in remaining_letters.values()):
                result = '\n'.join([' '.join(line) for line in haiku_lines])
                # Verify it's actually different from original
                if result.lower().replace(' ', '').replace('\n', '') != original_text.lower().replace(' ', '').replace('\n', ''):
                    return result
        
        return None
    
    def build_line(self, words_by_syllables: dict, available_letters: Counter, 
                   target_syllables: int) -> Optional[List[str]]:
        """
        Try to build a line with exact syllable count using available letters.
        Returns list of words or None if impossible.
        """
        def can_use_word(word: str, letters: Counter) -> bool:
            word_count = Counter(word.lower())
            return all(letters[char] >= count for char, count in word_count.items())
        
        def try_combination(remaining_syllables: int, current_words: List[str], 
                          current_letters: Counter) -> Optional[List[str]]:
            if remaining_syllables == 0:
                return current_words
            if remaining_syllables < 0:
                return None
            
            # Try words with different syllable counts
            for syll_count in range(min(remaining_syllables, 5), 0, -1):
                if syll_count not in words_by_syllables:
                    continue
                    
                for word in words_by_syllables[syll_count]:
                    if can_use_word(word, current_letters):
                        # Try using this word
                        new_letters = current_letters.copy()
                        for char in word.lower():
                            if char.isalpha():
                                new_letters[char] -= 1
                        
                        result = try_combination(
                            remaining_syllables - syll_count,
                            current_words + [word],
                            new_letters
                        )
                        
                        if result is not None:
                            return result
            
            return None
        
        # Try to build the line
        return try_combination(target_syllables, [], available_letters)


def main():
    print("=== Haiku Anagram Validator & Generator ===\n")
    print("Enter your haiku (3 lines, press Enter after each line):")
    print("Line 1 (5 syllables):")
    line1 = input().strip()
    print("Line 2 (7 syllables):")
    line2 = input().strip()
    print("Line 3 (5 syllables):")
    line3 = input().strip()
    
    haiku_text = f"{line1}\n{line2}\n{line3}"
    
    # Create analyzer
    analyzer = HaikuAnagrammer()
    
    # Validate input haiku
    is_valid, syllable_counts = analyzer.validate_haiku(haiku_text)
    
    if is_valid:
        print("✓ Valid haiku!")
    else:
        print("✗ Not a valid haiku.")
        print("Continuing anyway...")
    
    # Show letter counts
    letter_counts = analyzer.get_letter_counts(haiku_text)
    total_letters = sum(letter_counts.values())
    print(f"\nTotal letters: {total_letters}")
    print(f"Letter distribution: {dict(letter_counts.most_common())}")
    
    # Attempt to generate anagram haiku
    print("\n--- Searching for Anagram Haiku ---")
    print("This may take a while...")
    
    anagram_haiku = analyzer.generate_haiku_anagrams(haiku_text, max_attempts=1000000)
    
    if anagram_haiku:
        print("\n✓ Found an anagram haiku!")
        print("\nOriginal:")
        print(haiku_text)
        print("\nAnagram:")
        print(anagram_haiku)
        
        # Verify the anagram
        original_letters = analyzer.get_letter_counts(haiku_text)
        anagram_letters = analyzer.get_letter_counts(anagram_haiku)
        
        if original_letters == anagram_letters:
            print("\n✓ Verified: Uses exactly the same letters!")
        else:
            print("\n✗ Warning: Letter mismatch detected")
    else:
        print("\n✗ No valid haiku anagram exists for this input.")


if __name__ == "__main__":
    main()