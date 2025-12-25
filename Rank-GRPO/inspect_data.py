import pickle
import sys

file_path = "./SASRec_Data/sasrec_dataset.pkl"

print(f"🧐 Inspecting {file_path} ...")

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Data Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"🔑 Keys found: {list(data.keys())}")
        # 打印前几个 Key 的数据类型，帮助判断
        for k in list(data.keys())[:5]:
            print(f"   - {k}: {type(data[k])}")
            if hasattr(data[k], '__len__'):
                print(f"     Length: {len(data[k])}")
    elif isinstance(data, (list, tuple)):
        print(f"📏 Sequence Length: {len(data)}")
        print("   Content types:")
        for i, item in enumerate(data):
            print(f"   - Index {i}: {type(item)}")
    else:
        print(f"❓ Unexpected type: {data}")

except Exception as e:
    print(f"❌ Error loading file: {e}")