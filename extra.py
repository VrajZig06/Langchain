import ssl, certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())
print(ssl_context)