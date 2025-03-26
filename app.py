import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Veri ve model yükleme
@st.cache_data
def load_data():
    data = pd.read_csv('dating_app_dataset.csv')
    
    # Ön işleme adımları
    data['Interests_Count'] = data['Interests'].apply(lambda x: len(eval(x)))
    data['Height'] = data['Height'].apply(lambda x: round(x*30.48))
    
    def calculate_activity_score(row):
        usage_map = {'Daily': 3, 'Weekly': 2, 'Monthly': 1}
        return (row['Swiping History'] * 0.1) + (usage_map[row['Frequency of Usage']] * 10)
    
    data['Activity_Score'] = data.apply(calculate_activity_score, axis=1)
    
    scaler = joblib.load('scaler.pkl')
    numeric_features = ['Age', 'Height', 'Swiping History', 'Interests_Count', 'Activity_Score']
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    
    return data, scaler

data, scaler = load_data()

# EŞLEŞME SKOR FONKSİYONU
def enhanced_match_score(profile1, profile2):
    interests1 = set(eval(profile1['Interests']))
    interests2 = set(eval(profile2['Interests']))
    jaccard_similarity = len(interests1 & interests2) / len(interests1 | interests2)
    
    age_diff = abs(profile1['Age'] - profile2['Age'])
    height_diff = abs(profile1['Height'] - profile2['Height'])
    
    activity_compatibility = 1 - abs(profile1['Activity_Score'] - profile2['Activity_Score'])
    relationship_score = 2 if profile1['Looking For'] == profile2['Looking For'] else 0.5
    children_score = 2 if profile1['Children'] == profile2['Children'] else 0.8
    
    score = (jaccard_similarity * 40 + 
            (1 - age_diff) * 20 + 
            (1 - height_diff/50) * 15 + 
            activity_compatibility * 15 + 
            relationship_score * 5 + 
            children_score * 5)
    
    return round(score, 2)

# ÖNERİ SİSTEMİ FONKSİYONU
def optimized_recommendation(target_user, user_pool, top_n=5):
    scores = []
    for _, user in user_pool.iterrows():
        if user['User ID'] != target_user['User ID']:
            score = enhanced_match_score(target_user, user)
            scores.append((user.to_dict(), score))  # Pandas Series'i dict'e çeviriyoruz
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

# Özellik geri dönüşüm fonksiyonu
def inverse_transform_features(df):
    numeric_features = ['Age', 'Height', 'Swiping History', 'Interests_Count', 'Activity_Score']
    df[numeric_features] = scaler.inverse_transform(df[numeric_features])
    return df

# STREAMLIT ARAYÜZ
st.title('📱 Dating App Öneri Sistemi')
st.markdown("---")

valid_ids = data['User ID'].unique()
user_id = st.selectbox('🔍 Lütfen Profil ID\'nizi Seçin', valid_ids)
gender_preference = st.selectbox('🎯 İlgilendiğiniz Cinsiyet', ['Male', 'Female'])

if st.button('💘 Önerileri Göster', use_container_width=True):
    try:
        target_user = data[data['User ID'] == user_id].iloc[0]
        user_pool = data[data['Gender'] == gender_preference]
        
        if user_pool.empty:
            st.error("⚠️ Bu kriterlere uygun kullanıcı bulunamadı!")
        else:
            recommendations = optimized_recommendation(target_user, user_pool)
            
            st.success(f"✅ {user_id} ID'li kullanıcı için en iyi {len(recommendations)} öneri:")
            
            for idx, (profile, score) in enumerate(recommendations):
                profile_df = pd.DataFrame([profile])
                profile_df = inverse_transform_features(profile_df).iloc[0]
                
                with st.container():
                    cols = st.columns([1,3])
                    with cols[0]:
                        st.metric(label="**Eşleşme Skoru**", value=f"{score}%")
                        st.image("https://cdn-icons-png.flaticon.com/512/1077/1077114.png", width=100)
                    with cols[1]:
                        st.markdown(f"""
                        **📛 Profil ID:** {profile_df['User ID']}  
                        **🎂 Yaş:** {int(profile_df['Age'])}  
                        **📏 Boy:** {int(profile_df['Height'])} cm  
                        **🎓 Eğitim:** {profile_df['Education Level']}  
                        **❤️ İlişki Hedefi:** {profile_df['Looking For']}
                        """)
                        st.markdown(f"**🎯 İlgi Alanları:** {profile_df['Interests']}")
                    st.markdown("---")
                    
    except Exception as e:
        st.error(f"❌ Kritik Hata: {str(e)}")
        st.error("Lütfen veri setini ve model dosyalarını kontrol edin")