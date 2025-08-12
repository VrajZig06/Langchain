from mcp.server.fastmcp import FastMCP
import requests

MCPServer2 = FastMCP("CurrencyConvertor")

API_KEY = "e4155a91db7442daf47d9055"

@MCPServer2.tool()
async def convert_currency(base_country_code:str,target_country_code:str,amount:int):
    """
    This is App Where you convert your currency from base country to target country
    """
    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/pair/{base_country_code}/{target_country_code}"

    response = await requests.get(url)
    conversion_rate = response.json()['conversion_rate']

    result = conversion_rate * amount

    return result


if __name__ == "__main__":
    MCPServer2.run(transport = "streamable-http")

