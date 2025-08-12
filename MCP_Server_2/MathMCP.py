"""
    To Make any MCP Server we need two main libraries:
    1. Langgraph
    2. langchain-mcp-adapter

    Transport: 
    - stdio (Standard Input/Output)
    - streamable-http (used when we need to work using API)


"""



from mcp.server.fastmcp import FastMCP

MCPServer = FastMCP("Math_MCP")

@MCPServer.tool()
def add_two_numbers(a: int, b: int) -> int:
    """
    Add two integers and return the sum.

    This function takes two integer inputs, adds them together,
    and returns the resulting sum.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of `a` and `b`.

    Example:
        >>> add_two_numbers(3, 5)
        8
    """
    return a + b


@MCPServer.tool()
def multiply_two_numbers(a:int,b:int) -> int:
    """
        This is Function that is used to multiply two numbers.

        ex. a = 1 , b = 2. 
            Then return a * b = 1 * 2 = 2

    """
    return a*b

if __name__ == "__main__":
    MCPServer.run(transport = "stdio")