import wbgapi as wb
import pandas as pd

# Try using wb.data.DataFrame
try:
    result = wb.data.DataFrame("SP.DYN.LE00.IN", time=range(2020, 2024))
    print("Successfully fetched with wb.data.DataFrame()")
    print(type(result))
    print(result.shape)
    print(result.head())
except Exception as e:
    print(f"Error with DataFrame: {e}")

print("\n" + "="*60 + "\n")

# Try using wb.data.FlatFrame
try:
    result = wb.data.FlatFrame("SP.DYN.LE00.IN", time=range(2020, 2024))
    print ("Successfully fetched with wb.data.FlatFrame()")
    print(type(result))
    print(result.shape)
    print(result.head(10))
except Exception as e:
    print(f"Error with FlatFrame: {e}")

