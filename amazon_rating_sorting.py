

#### GEREKLİ KÜTÜPHANELERİ İMPORT EDELİM:

import pandas as pd
import math
import scipy.stats as st
import datetime as dt


#### EN ÇOK YORUM ALAN ÜRÜNE GÖRE:

df = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/df_sub.csv" )
df.head()
df.columns
pd.set_option('display.max_columns', None)
df.head()

#### ÜRÜNÜN ORTALAMA PUANI:

df["overall"].mean()

#Tarihe ağırlıklı puan ortalaması:

#1)day_diff hesaplamak için: (yorum sonrası ne kadar gün geçmiş)
df['reviewTime'] = pd.to_datetime(df['reviewTime'], dayfirst=True)
current_date = pd.to_datetime('2014-12-08 0:0:0')
df["day_diff"] = (current_date - df['reviewTime']).dt.days

#2)Zamanı çeyrek değerlere göre bölme:
a = df["day_diff"].quantile(0.25)
b = df["day_diff"].quantile(0.50)
c = df["day_diff"].quantile(0.75)

df.head()

#### a,b,c DEĞERLERİNE GÖRE AĞIRLIKLI PUAN ORTALAMASI HESAPLANMASI

df.loc[df["day_diff"] <= a, "overall"].mean() * 28 / 100 + \
df.loc[(df["day_diff"] > a) & (df["day_diff"] <= b), "overall"].mean() * 26 / 100 + \
df.loc[(df["day_diff"] > b) & (df["day_diff"] <= c), "overall"].mean() * 24 / 100 + \
df.loc[(df["day_diff"] > c), "overall"].mean() * 22 / 100



#### HELPFUL DEĞİŞKENİ İÇERİSİNDEN HELPFUL_YES, HELPFUL TOTAL'İ ÇEKMEK İÇİN FOR DÖNGÜSÜ KULLANALIM:

# Helpful içerisinde 2 değer vardır. Birincisi yorumları faydalı bulan oy sayısı ikincisi toplam oy sayısı.
# Dolayısıyla önce ikisini ayrı ayrı çekmeli sonra da (total_vote - helpful_yes) yaparak helpful_no'yu hesaplamalıyız.

df.head()
df.info()
df["helpful"].head()
df["helpful"].sort_values(ascending=False).head(10)

j=0
df["helpful_yes"]=""
df["total_vote"]=""
for i in df.helpful:
    print(type(i))
    i=i.rstrip("]")
    i=i.lstrip("[")
    i=i.split(",")
    df["helpful_yes"][j]= int(i[0])
    df["total_vote"][j] = int(i[1])
    j =j+1

df["helpful_yes"].head(20)
df["total_vote"].head(20)

#### HELPFUL_NO'YU HESAPLAYARAK DEĞİŞKEN OLUŞTURALIM.
df["helpful_no"]=df["total_vote"]-df["helpful_yes"]
df["helpful_no"].head(20)


#### SCORE_POS_NEG_DİFF'A GÖRE SKORLAR OLUŞTURUP VE DF_SUB İÇERİSİNDE SCORE_POS_NEG_DİFF İSMİYLE KAYDEDELİM:

def score_pos_neg_diff(pos, neg):
    return pos - neg
df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df["score_pos_neg_diff"].sort_values(ascending=False)


#### SCORE_AVERAGE_RATİNG'A GÖRE SKORLAR OLUŞTURUP VE DF_SUB İÇERİSİNDE SCORE_AVERAGE_RATİNG İSMİYLE KAYDEDELİM:

def score_average_rating(pos, neg):
    if pos + neg == 0:
        return 0
    return pos / (pos + neg)
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

df["score_average_rating"].sort_values(ascending=False)


#### WİLSON_LOWER_BOUND'A GÖRE SKORLAR OLUŞTURUP VE DF_SUB İÇERİSİNDE WİLSON_LOWER_BOUND İSMİYLE KAYDEDELİM:

def wilson_lower_bound(pos, neg, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not: Eğer skorlar 1-5 arasıdaysa 1-3 down, 4-5 up olarak işaretlenir ve bernoulli'ye uygun hale getirilir.

    Parameters
    ----------
    pos: int
        pozitif yorum sayısı
    neg: int
        negatif yorum sayısı
    confidence: float
        güven aralığı

    Returns
    -------
    wilson score: float

    """
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)


#### ÜRÜN SAYFASINDA GÖSTERİLECEK 20 YORUMU BELİRLEYİNİZ VE SONUÇLARI YORUMLAYINIZ.


df.sort_values("wilson_lower_bound", ascending=False).head(20)
df.sort_values("score_average_rating", ascending=False).head(20)
df.sort_values("score_pos_neg_diff", ascending=False).head(20)

#2031 numaralı yorum en yüksek score_neg_diff ve wilson_lower_bound değerlerine sahip.
#bu yorumu faydalı olarak gören kişi sayısı 1952 faydasız gören kişi sayısı ise 68 yani 1952'ye göre oldukça düşük bir oran sonuçlarla paralel.
#score_neg_diff ve wilson_lower_bound sıralamalarında 3449 ve 4212 numaralı ürünlerin sıralaması değişmiş. Bunun sebebi:
#score_neg_diff hesaplarken yorumların pozitif ve negatif değerlendirilmelerinin farkını almamız. Wilson testinde ise güven aralığı kullanarak yorumun faydasını ölçmemiz.
#4672 numaralı yorumu beğenen sayısı 45, 1835 numaralı ürünü beğenen sayısı ise 60. 1835 numaralı yorumun beğeni sayısı daha fazla olduğu halde
#Wilson lower bound'da daha aşağıda yer almakta. 4672 numaralı yorumda 45 beğenide 4 beğenmeyen varken, 1835 numaralı yorumda 60 beğenide ise 8 beğenmeyen mevcut.
#12. sıra ve sonrasında 4302 numaralı yorum hariç, yorumları faydalı bulmayan olmadığı halde bu yorumların sıralamada daha aşağıda olma sebebi yorum sayısının az olması.
# Çünkü Wilson lower bound'a göre hesaplama yaparken mantıklı ve adaletli bir dağılım olması için yorumların sayısı da göz önünde bulunduruluyor.
