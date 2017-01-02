import pandas as pd
import numpy as np

data_path = "./"
#data_path = "data/"

#document_id(display_id from event.csv),document_id_of_ad_id(ad_id from promoted_content.csv),campaign id,ad_count,leak

def run():
    clicks_train = pd.read_csv(data_path+'clicks_train.csv',
                               usecols=['display_id', 'ad_id', 'clicked'])
    events = pd.read_csv(data_path+'events.csv', 
                         usecols=['display_id', 'document_id'])
    promoted_content = pd.read_csv(data_path+'promoted_content.csv', 
                                   usecols=['ad_id','document_id', 'campaign_id'] )
    leak = pd.read_csv(data_path+'leak.csv', 
                       usecols=['document_id', 'uuid'] )
    print 'clicks_train:', clicks_train.shape
    print clicks_train.head()
    print 'events:', events.shape
    print events.head()
    print 'promoted_content:', promoted_content.shape
    print promoted_content.head()

    df1 = pd.merge(clicks_train, events, on='display_id', how='left')
    print 'df1', df1.shape
    print df1.head()
    df1 = pd.merge(df1, promoted_content, on='ad_id', how='left')
    df1.columns = ['display_id', 'ad_id', 'clicked', 'document_id', 'document_id_of_ad_id', 'campaign_id']
    print 'df1', df1.shape
    print df1.head()


    features = ['document_id', 'document_id_of_ad_id', 'campaign_id', 'ad_count', 'leak']
    data = { feature:[] for feature in features }
    features_train = pd.DataFrame(data, columns = features)
    features_train.to_csv('features_train.csv', index=False)

if __name__ == "__main__":
    run()
