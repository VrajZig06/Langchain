from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

text = """
                def is_prime(n):
                    if n <= 1:
                        return False
                    for i in range(2, int(n**0.5)+1):
                        if n % i == 0:
                            return False
                    return True

                # Example usage
                number = int(input("Enter a number: "))
                if is_prime(number):
                    print(f"{number} is a prime number.")
                else:
                    print(f"{number} is not a prime number.")
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 20,
    chunk_overlap = 0
)

result = splitter.split_text(text)

print(result)