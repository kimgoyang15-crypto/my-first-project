import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pydeck as pdk

st.set_page_config(page_title="Okcheon Housing Dashboard", layout="wide")

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

@st.cache_data
def load_data():
    df_house = pd.read_csv('충청북도 옥천군_공동주택 현황_20260915.csv', encoding='cp949')
    df_house['Region'] = df_house['소재지'].str.extract(r'(옥천읍|동이면|안남면|안내면|청성면|청산면|이원면|군서면|군북면)')
    
    np.random.seed(42)
    df_house['lat'] = 36.30 + np.random.randn(len(df_house)) * 0.03
    df_house['lon'] = 127.57 + np.random.randn(len(df_house)) * 0.03

    def safe_read_csv(filename):
        try:
            return pd.read_csv(filename, encoding='cp949')
        except:
            try:
                return pd.read_csv(filename, encoding='utf-8')
            except:
                return pd.read_csv(filename, encoding='euc-kr')

    df_school = safe_read_csv('한국교육시설안전원_초중등학교위치.csv')
    df_hosp = safe_read_csv('충청북도 옥천군_의료기관현황_20260826.csv')
    df_pharm = safe_read_csv('충청북도 옥천군_약국현황_20260826.csv')

    def clean_lat_lon(df):
        lat_col = next((c for c in df.columns if '위도' in c or 'lat' in c.lower()), None)
        lon_col = next((c for c in df.columns if '경도' in c or 'lon' in c.lower() or 'lng' in c.lower()), None)
        
        if lat_col and lon_col:
            df['lat'] = pd.to_numeric(df[lat_col], errors='coerce')
            df['lon'] = pd.to_numeric(df[lon_col], errors='coerce')
            df = df.dropna(subset=['lat', 'lon'])
        else:
            df['lat'] = 36.30 + np.random.randn(len(df)) * 0.03
            df['lon'] = 127.57 + np.random.randn(len(df)) * 0.03
        return df

    df_school = clean_lat_lon(df_school)
    df_hosp = clean_lat_lon(df_hosp)
    df_pharm = clean_lat_lon(df_pharm)

    def calc_min_distance(row, target_df):
        if target_df.empty:
            return 1.0
        dists = np.sqrt((target_df['lat'] - row['lat'])**2 + (target_df['lon'] - row['lon'])**2) * 111
        return round(float(dists.min()), 2)

    df_house['School_Dist'] = df_house.apply(lambda r: calc_min_distance(r, df_school), axis=1)
    df_house['Hosp_Dist'] = df_house.apply(lambda r: calc_min_distance(r, df_hosp), axis=1)
    df_house['Pharm_Dist'] = df_house.apply(lambda r: calc_min_distance(r, df_pharm), axis=1)

    max_household = df_house['세대수'].max() if df_house['세대수'].max() > 0 else 1
    max_floor = df_house['층수'].max() if df_house['층수'].max() > 0 else 1
    df_house['Basic_Score'] = (df_house['세대수'] / max_household * 50) + (df_house['층수'] / max_floor * 50)
    
    max_school = df_house['School_Dist'].max() if df_house['School_Dist'].max() > 0 else 1
    max_hosp = df_house['Hosp_Dist'].max() if df_house['Hosp_Dist'].max() > 0 else 1
    max_pharm = df_house['Pharm_Dist'].max() if df_house['Pharm_Dist'].max() > 0 else 1
    
    df_house['Dist_Score'] = (
        (1 - df_house['School_Dist'] / max_school) * 33.3 +
        (1 - df_house['Hosp_Dist'] / max_hosp) * 33.3 +
        (1 - df_house['Pharm_Dist'] / max_pharm) * 33.4
    )
    
    df_house['Total_Score'] = (df_house['Basic_Score'] * 0.4) + (df_house['Dist_Score'] * 0.6)
    
    return df_house

df_house = load_data()

st.title("옥천군 읍·면별 공동주택 지도 및 취약지역 분석 대시보드")
st.markdown("---")

regions = ['전체'] + list(df_house['Region'].dropna().unique())
selected_region = st.selectbox("조회할 읍·면을 선택하세요", regions)

if selected_region != '전체':
    filtered_df = df_house[df_house['Region'] == selected_region]
else:
    filtered_df = df_house

st.markdown("### 주요 통계 요약 지표")
c1, c2, c3, c4 = st.columns(4)
c1.metric("총 단지 수", f"{len(filtered_df):,}개")
c2.metric("총 세대수", f"{filtered_df['세대수'].sum():,}세대")
c3.metric("평균 세대수", f"{filtered_df['세대수'].mean():.1f}세대")
c4.metric("평균 종합생활편의 점수", f"{filtered_df['Total_Score'].mean():.1f}점")

st.markdown("---")

st.markdown("### 단지별 상세 인프라 거리 및 편의 지표 패널")
if not filtered_df.empty:
    sorted_filtered_df = filtered_df.sort_values(by='Total_Score', ascending=False)
    
    selected_complex = st.selectbox("상세 지표를 조회할 단지를 선택하세요 (종합편의점수 높은 순)", sorted_filtered_df['단지명'].unique())
    info = sorted_filtered_df[sorted_filtered_df['단지명'] == selected_complex].iloc[0]
    
    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
    sc1.metric("세대수", f"{info['세대수']}세대")
    sc2.metric("학교 거리", f"{info['School_Dist']} km")
    sc3.metric("병원 거리", f"{info['Hosp_Dist']} km")
    sc4.metric("약국 거리", f"{info['Pharm_Dist']} km")
    sc5.metric("기본 편의점수", f"{info['Basic_Score']:.1f}점")
    sc6.metric("종합편의점수", f"{info['Total_Score']:.1f}점")

st.markdown("---")

st.markdown("### 옥천군 공동주택 단지 위치 지도")
layer = pdk.Layer(
    "ScatterplotLayer",
    data=filtered_df,
    get_position='[lon, lat]',
    get_color='[200, 30, 0, 160]',
    get_radius='세대수 * 2',
    pickable=True,
    auto_highlight=True,
)

view_state = pdk.ViewState(
    latitude=filtered_df['lat'].mean() if not filtered_df.empty else 36.30,
    longitude=filtered_df['lon'].mean() if not filtered_df.empty else 127.57,
    zoom=11,
    pitch=0
)

tooltip_config = {
    "html": "<b>단지명:</b> {단지명} <br/><b>소재지:</b> {소재지} <br/><b>세대수:</b> {세대수}세대",
    "style": {"backgroundColor": "steelblue", "color": "white", "zIndex": "10000"}
}

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip_config))

st.markdown("---")

st.markdown("### 읍·면별 인구 대비 공동주택 공급 비교 분석")
region_house_sum = df_house.groupby('Region')['세대수'].sum().reset_index()
region_house_sum.columns = ['Region', 'House_Sum']

all_regions_list = ['옥천읍', '동이면', '안남면', '안내면', '청성면', '청산면', '이원면', '군서면', '군북면']
analysis_df = pd.DataFrame({'Region': all_regions_list})
analysis_df = pd.merge(analysis_df, region_house_sum, on='Region', how='left').fillna(0)

pop_dict = {'옥천읍': 29000, '이원면': 4500, '청산면': 2500, '군북면': 3000, '군서면': 2800, 
            '동이면': 3200, '안남면': 1500, '안내면': 1600, '청성면': 1800}
analysis_df['Population'] = analysis_df['Region'].map(pop_dict)
analysis_df['Ratio'] = (analysis_df['House_Sum'] / analysis_df['Population']) * 100

col_l, col_r = st.columns(2)
with col_l:
    st.markdown("#### 읍·면별 인구 100명당 공동주택 세대수")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=analysis_df, x='Region', y='Ratio', palette='Reds_r', ax=ax)
    ax.set_xlabel("읍·면")
    ax.set_ylabel("인구 100명당 세대수")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col_r:
    st.markdown("#### 읍·면별 주거 취약 지역 진단 표")
    st.dataframe(analysis_df.sort_values(by='Ratio', ascending=True).reset_index(drop=True), use_container_width=True)

st.markdown("---")

left_col, right_col = st.columns(2)
with left_col:
    st.markdown("### 읍·면별 단지 수 비교")
    region_counts = df_house['Region'].value_counts().reset_index()
    region_counts.columns = ['Region', 'Count']
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=region_counts, x='Region', y='Count', palette='Blues_r', ax=ax2)
    ax2.set_xlabel("읍·면")
    ax2.set_ylabel("단지 수")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

with right_col:
    st.markdown("### 생활 편의성 상위 5개 단지 정보")
    top5_df = filtered_df.sort_values(by='Total_Score', ascending=False).head(5)
    display_cols = ['단지명', '소재지', '세대수', 'School_Dist', 'Hosp_Dist', 'Pharm_Dist', 'Total_Score']
    existing_cols = [c for c in display_cols if c in top5_df.columns]
    
    top5_display = top5_df[existing_cols].copy()
    if 'Total_Score' in top5_display.columns:
        top5_display['Total_Score'] = top5_display['Total_Score'].round(1)
        
    st.dataframe(top5_display.reset_index(drop=True), use_container_width=True)