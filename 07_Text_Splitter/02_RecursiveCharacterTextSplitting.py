from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """Cricket is one of the most popular sports in the world, especially in countries like India, Australia, and England. It is played between two teams of eleven players each on a large field with a 22-yard pitch at the center. The game has three main formats: Test, One Day International (ODI), and Twenty20 (T20). Test matches are the longest format, lasting up to five days, while T20 is the shortest and most fast-paced. The Indian Premier League (IPL) is one of the most lucrative and widely watched T20 leagues. Legendary players like Sachin Tendulkar, M.S. Dhoni, and Virat Kohli have brought immense pride to Indian cricket. Bowlers like Muttiah Muralitharan and Shane Warne changed the dynamics of spin bowling. The game is governed internationally by the ICC (International Cricket Council). Technology like DRS (Decision Review System) has enhanced the fairness of umpiring decisions. Cricket continues to evolve, gaining fans and adapting to modern-day entertainment.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 20,
    chunk_overlap = 0
)

result = splitter.split_text(text)

print(result)