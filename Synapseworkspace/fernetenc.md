# Apply fernet encryption with Azure Synapse Spark

## Protect your PII by encrypting using Fernet in Azure Synapse Spark

## Using fernet to encrypt key - symmetric encryption

## Prerequistie

- Azure Account
- Azure Storage
- Azure Synaspe Analytics workspace with Spark

## Use Case

- Use encryption to encrypt PII or other sensitive data
- Data should be stored encrypted
- Only folks who have access to key can decrypt
- encrypt column level so only necessary columns can be encrypted and other's are available for reporting
- here is the open source encryption project - https://cryptography.io/en/latest/fernet/

## Code

- First create a spark cluster
- install cryptography library
- Create environment.yaml file

```
name: stats
dependencies:
  - numpy
  - pandas
  - cryptography
```

- wait for the package to install
- Create a new notebook
- Add the package yml file

```
from cryptography.fernet import Fernet
# >>> Put this somewhere safe!
key = Fernet.generate_key()
```

- print and see

```
f = Fernet(key)
token = f.encrypt(b"A really secret message. Not for prying eyes.")
print(token)
print(f.decrypt(token))
```

- Create the UDF for encrypt and decrypt

```
# Define Encrypt User Defined Function 
def encrypt_val(clear_text,MASTER_KEY):
    from cryptography.fernet import Fernet
    f = Fernet(MASTER_KEY)
    clear_text_b=bytes(clear_text, 'utf-8')
    cipher_text = f.encrypt(clear_text_b)
    cipher_text = str(cipher_text.decode('ascii'))
    return cipher_text

# Define decrypt user defined function 
def decrypt_val(cipher_text,MASTER_KEY):
    from cryptography.fernet import Fernet
    f = Fernet(MASTER_KEY)
    clear_val=f.decrypt(cipher_text.encode()).decode()
    return clear_val
```

- read data

```
df = spark.read.load('abfss://containername@storagename.dfs.core.windows.net/titanic/Titanic.csv', format='csv'
## If header exists uncomment line below
, header=True
)

print(df)
```

- encrypt data

```
from pyspark.sql.functions import udf, lit, md5
from pyspark.sql.types import StringType

# Register UDF's
encrypt = udf(encrypt_val, StringType())
decrypt = udf(decrypt_val, StringType())

# Fetch key from secrets
# encryptionKey = dbutils.preview.secret.get(scope = "encrypt", key = "fernetkey")
encryptionKey = key

# Encrypt the data 
#df = spark.table("Test_Encryption")
encrypted = df.withColumn("Name", encrypt("Name",lit(encryptionKey)))
encrypted.head()
```

- decrypt test

```
decrypted = encrypted.withColumn("Name", decrypt("Name",lit(encryptionKey)))
decrypted.head()
```