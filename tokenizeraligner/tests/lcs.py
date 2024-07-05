def longest_common_subsequence(seq1, seq2):
    m, n = len(seq1), len(seq2)
    # Create a 2D array to store lengths of longest common subsequence
    L = [[0] * (n + 1) for _ in range(m + 1)]

    # Build the L[m+1][n+1] table in bottom-up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif seq1[i - 1] == seq2[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # Following code is used to print LCS
    index = L[m][n]
    
    # Create a list to store the LCS sequence
    lcs = [None] * index
    i, j = m, n

    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs[]
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            lcs[index - 1] = seq1[i - 1]
            i -= 1
            j -= 1
            index -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return lcs

# Example usage
seq1 = [90, 1, 3, 4, 9]
seq2 = [80, 1, 3, 5, 5, 9]

common_sequence = longest_common_subsequence(seq1, seq2)
print("Longest Common Subsequence:", common_sequence)
print('angela')
