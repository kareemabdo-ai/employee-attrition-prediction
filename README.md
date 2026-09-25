# 🏢 Employee Attrition Prediction

مشروع Machine Learning للتنبؤ باحتمالية ترك الموظفين للشركة، بناءً على بيانات الـ HR.

## 📊 نظرة عامة
- **الداتاست:** IBM HR Analytics Employee Attrition (Kaggle)
- **نوع المشكلة:** Binary Classification
- **التحدي الأساسي:** Class Imbalance (نسبة "Yes" حوالي 16% بس من البيانات)

## 🔧 الخطوات
1. تنظيف البيانات وإزالة الأعمدة الغير مفيدة (`EmployeeNumber`, `EmployeeCount`, `StandardHours`, `Over18`)
2. Encoding (One-Hot للأعمدة متعددة القيم + Label Encoding للأعمدة الثنائية)
3. تجربة Logistic Regression و Random Forest و XGBoost
4. معالجة الـ Class Imbalance عبر `class_weight` وضبط الـ decision threshold
5. اختيار أفضل موديل بناءً على Recall/F1 مش Accuracy بس

## 🏆 أفضل موديل
**Random Forest (threshold = 0.2)**

| Metric | Value |
|---|---|
| Recall (Yes) | 0.64 |
| Precision (Yes) | 0.41 |
| F1-Score (Yes) | 0.50 |
| Accuracy | 0.79 |

## 🚀 تشغيل الـ App محليًا
```bash
pip install -r requirements.txt
cd app
streamlit run app.py
```

## 📁 هيكل المشروع
```
├── notebook/    # التحليل الكامل والتدريب (Colab notebook)
├── model/       # الموديل المحفوظ (attrition_model.pkl, model_columns.pkl)
└── app/         # واجهة Streamlit (app.py)
```

## ⚠️ ملاحظة قبل التشغيل
لازم تحط في فولدر `model/` الملفين الناتجين من الـ notebook:
- `attrition_model.pkl`
- `model_columns.pkl`

وتحط الـ notebook بتاعك في فولدر `notebook/`.
