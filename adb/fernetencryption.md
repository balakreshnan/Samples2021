# Azure databricks encryption

## Using fernet to encrypt key - symmetric encryption

## Prerequistie

- Azure Account
- Azure Storage
- Azure databricks

## Use Case

- Use encryption to encrypt PII or other sensitive data
- Data should be stored encrypted
- Only folks who have access to key can decrypt
- encrypt column level so only necessary columns can be encrypted and other's are available for reporting
- here is the open source encryption project - https://cryptography.io/en/latest/fernet/

## Code

- First create a databricks cluster
- install cryptography library
- After cluster starts go to library
- Select pypi and type: cryptography and install
- wait for the package to install
- Create a new notebook

```
%sql 
use default; -- Change this value to some other database if you do not want to use the Databricks default

drop table if exists Test_Encryption;

create table Test_Encryption(Name string, Address string, ssn string) USING DELTA;
```

```
%sql
insert into Test_Encryption values ('Mark Smith', 'my street, universe', '6789454');
insert into Test_Encryption values ('King Solomon', 'somewhere in earth, Artic', '98023456');
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/fernet1.jpg "Service Health")

- Sample code to test the encryption library

```
from cryptography.fernet import Fernet
# >>> Put this somewhere safe!
key = Fernet.generate_key()
f = Fernet(key)
token = f.encrypt(b"A really secret message. Not for prying eyes.")
print(token)
print(f.decrypt(token))
```

- Now create UDF for encrypt and decrypt

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

- now lets encrypt a column

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
df = spark.table("Test_Encryption")
encrypted = df.withColumn("ssn", encrypt("ssn",lit(encryptionKey)))
display(encrypted)

#Save encrypted data 
encrypted.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("Test_Encryption_Table")
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/fernet2.jpg "Service Health")

- now decrypt

```
decrypted = encrypted.withColumn("ssn", decrypt("ssn",lit(encryptionKey)))
display(decrypted)
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/fernet3.jpg "Service Health")