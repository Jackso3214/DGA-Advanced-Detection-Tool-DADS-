#vowel and consonants counter

def VowelConsonants(string):

    string = string.lower()
    #create a counter dict for consonants and vowels
    counts = {"vowel": 0, "consonant": 0}
    #list of vowels to check
    vowels = ['a', 'e', 'i', 'o', 'u']

    #check each character to see if in vowel or consonants
    for character in string:
        if character in vowels:
            counts["vowel"] += 1
        elif character.isalpha():
            counts["consonant"] += 1
    return counts