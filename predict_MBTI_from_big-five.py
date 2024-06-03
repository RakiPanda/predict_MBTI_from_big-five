import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# CSVファイルを読み込む
data = pd.read_csv('final.csv')

# MBTIの閾値を定義
thresholds = {
    'extraverted': 50,
    'intuitive': 50,
    'thinking': 50,
    'judging': 50
}

# 閾値に基づいてMBTIラベルを定義
mbti_labels = {
    'extraverted': {False: 'I', True: 'E'},
    'intuitive': {False: 'S', True: 'N'},
    'thinking': {False: 'F', True: 'T'},
    'judging': {False: 'P', True: 'J'}
}

# 各人のスコアに基づいてMBTIタイプを分類
data['MBTI'] = data.apply(lambda row: ''.join([mbti_labels[trait][row[trait] >= thresholds[trait]] 
                                               for trait in thresholds]), axis=1)

# 新しいCSVファイルとして結果を保存
data.to_csv('final_with_mbti.csv', index=False)

# 確認のため、新しいデータの最初の5行を出力
print(data[['id', 'extraverted', 'intuitive', 'thinking', 'judging', 'MBTI']].head())

# CSVファイルを読み込む
data = pd.read_csv('final_with_mbti.csv')

# MBTIの分布を集計
mbti_counts = data['MBTI'].value_counts()

# 分布を棒グラフとしてプロット
plt.figure(figsize=(12, 8))
sns.barplot(x=mbti_counts.index, y=mbti_counts.values, palette="viridis")
plt.title('MBTI Type Distribution')
plt.xlabel('MBTI Types')
plt.ylabel('Counts')
plt.xticks(rotation=45)  # MBTIのラベルが重ならないように回転

# グラフを画像ファイルとして保存
plt.savefig('mbti_distribution.png')

# グラフを表示
plt.show()


# CSVファイルを読み込む
data = pd.read_csv('final_with_mbti.csv')

# 特徴量とターゲットを定義
features = data[['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']]
target = data['MBTI']

# ラベルを数値にエンコード
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)

# ランダムフォレスト分類器をインスタンス化し、訓練
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# テストデータに対する予測
predictions = classifier.predict(X_test)

# 性能評価を行い、CSVに保存
report = classification_report(y_test, predictions, target_names=label_encoder.classes_, labels=np.arange(len(label_encoder.classes_)), zero_division=1, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv('classification_report.csv', index=True)

# 結果をコンソールにも出力
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report Saved to CSV.")