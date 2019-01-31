#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'long_tail_queries/data_and_train'))
	print(os.getcwd())
except:
	pass

#%%
TRAIN_CTR_PATH = "learning/CTRtrain"
TRAIN_sdbn_PATH = "learning/SDBNtrain"
TEST_CTR_PATH = "learning/CTRtest"
TEST_sdbn_PATH = "learning/SDBNtest"
MEANCTRTRAINPATH = "learning/MeanCTRtrain"
MEANCTRTESTPATH = "learning/MeanCTRtest"
QDPOSTRAINPATH = "learning/QDPosTrain"
QDPOSTESTPATH = "learning/QDPosTest"


#%%
import pandas as pd
import numpy as np
TRAINMARKSPATH = "behrank_data/train.marks"
SAMPLEPATH = "behrank_data/sample"
URLDATAPATH = "behrank_data/urldata"
QUERIESPATH = "behrank_data/queries"
trainmarks = pd.read_csv(TRAINMARKSPATH, sep="\t", header = None)
trainmarks.columns = ["qid", "uid", "mark"]
sample = pd.read_csv(SAMPLEPATH, sep=",", header = None)
sample.columns = ["qid", "uid"]
urldata = pd.read_csv(URLDATAPATH, sep="\t", header = None)
urldata.columns = ["uid", "url"]
querydata = pd.read_csv(QUERIESPATH, sep="\t", header = None)
querydata.columns = ["qid", "q"]


#%%
train_ctr = pd.read_csv(TRAIN_CTR_PATH, sep = "\t", header = None)
train_ctr.columns = ["quname", "result", "mark"]
tmp = train_ctr["quname"].str.split()
assert((train_ctr["quname"].apply(lambda x:"  " in x)==True).sum()==0)
train_ctr["q"] = tmp.apply(lambda x: " ".join(x[:-1]))
train_ctr["url"] = tmp.apply(lambda x: x[-1])
train_ctr = train_ctr.set_index("q").join(querydata.set_index("q")).set_index("url").join(urldata.set_index("url"))

test_ctr = pd.read_csv(TEST_CTR_PATH, sep = "\t", header = None)
test_ctr.columns = ["result", "quid"]
tmp = test_ctr["quid"].str.split(",")
test_ctr["qid"] = tmp.apply(lambda x:int(x[0]))
test_ctr["uid"] = tmp.apply(lambda x:int(x[1]))

train_qdp = pd.read_csv(QDPOSTRAINPATH, sep = "\t", header = None)
train_qdp.columns = ["quname", "result", "mark"]
tmp = train_qdp["quname"].str.split()
assert((train_qdp["quname"].apply(lambda x:"  " in x)==True).sum()==0)
train_qdp["q"] = tmp.apply(lambda x: " ".join(x[:-1]))
train_qdp["url"] = tmp.apply(lambda x: x[-1])
train_qdp = train_qdp.set_index("q").join(querydata.set_index("q")).set_index("url").join(urldata.set_index("url"))

test_qdp = pd.read_csv(QDPOSTESTPATH, sep = "\t", header = None)
test_qdp.columns = ["result", "quid"]
tmp = test_qdp["quid"].str.split(",")
test_qdp["qid"] = tmp.apply(lambda x:int(x[0]))
test_qdp["uid"] = tmp.apply(lambda x:int(x[1]))

train_sdbn = pd.read_csv(TRAIN_sdbn_PATH, sep="\t", header = None)
train_sdbn.columns = ["quname", "result", "mark"]
tmp = train_sdbn["quname"].str.split()
assert((train_sdbn.apply(lambda x:"  " in x)==True).sum()==0)
train_sdbn["q"] = tmp.apply(lambda x: " ".join(x[:-1]))
train_sdbn["url"] = tmp.apply(lambda x: x[-1])
train_sdbn = train_sdbn.set_index("q").join(querydata.set_index("q")).set_index("url").join(urldata.set_index("url"))

test_sdbn = pd.read_csv(TEST_sdbn_PATH, sep="\t", header = None)
test_sdbn.columns = ["result", "quid"]
tmp = test_sdbn["quid"].str.split(",")
test_sdbn["qid"] = tmp.apply(lambda x:int(x[0]))
test_sdbn["uid"] = tmp.apply(lambda x:int(x[1]))

train_meanctr = pd.read_csv(MEANCTRTRAINPATH, sep="\t", header = None)
train_meanctr.columns = ["url", "result"]
train_meanctr = pd.merge(train_meanctr, urldata,  how='left', left_on=["url"], right_on = ["url"])

test_meanctr = pd.read_csv(MEANCTRTESTPATH, sep="\t", header = None)
test_meanctr.columns = ["result", "uid"]


#%%
test_qdp.head()


#%%
def add_qdpos(df):
    result_to_list = lambda x: tuple([float(item) for item in x.split()])
    a = df["result"].apply(result_to_list)
    df["sh_mean"] = a.apply(lambda x: x[0])
    df["sh_cnt"] = a.apply(lambda x: x[1])
    df["sh_max"] = a.apply(lambda x: x[2])
    df["sh_min"] = a.apply(lambda x: x[3])
    
    df["cl_mean"] = a.apply(lambda x: x[4])
    df["cl_cnt"] = a.apply(lambda x: x[5])
    df["cl_max"] = a.apply(lambda x: x[6])
    df["cl_min"] = a.apply(lambda x: x[7])
    
    return df

train_qdp = add_qdpos(train_qdp)
test_qdp = add_qdpos(test_qdp)


#%%
train_qdp.head()


#%%
def calc_ctr(s):
    pair = s.split()
    return float(pair[1])/float(pair[0])
train_ctr["CTR"] = train_ctr["result"].apply(calc_ctr)
test_ctr["CTR"] = test_ctr["result"].apply(calc_ctr)


#%%
def calc_sdbn(df, index):
    alphaa = 0.2
    betaa = 0.2
    alphas = 0.2
    betas = 0.2
    result_to_list = lambda x: tuple([float(a) for a in x.split()])
    a = df[index].apply(result_to_list)
    df["an"] = a.apply(lambda x: x[0])
    df["ad"] = a.apply(lambda x: x[1])
    df["sn"] = a.apply(lambda x: x[2])
    df["sd"] = a.apply(lambda x: x[3])
    df["a"] = (df["an"] + alphaa)/(df["ad"] + alphaa + betaa)
    df["s"] = (df["sn"] + alphas)/(df["sd"] + alphas + betas)
    return df
train_sdbn = calc_sdbn(train_sdbn, index="result")
test_sdbn = calc_sdbn(test_sdbn, index="result")


#%%
def calc_mctr(df, index):
    a = df[index].str.split(" ")
    df["mctr0"] = a.apply(lambda x: float(x[0]))
    df["mctr1"] = a.apply(lambda x: float(x[1]))
    df["mctrgeom"] = np.sqrt(df.mctr0 * df.mctr1)
    return df
train_meanctr = calc_mctr(train_meanctr, "result")
test_meanctr = calc_mctr(test_meanctr, "result")


#%%
train = trainmarks.copy()
train = pd.merge(train, train_sdbn[["qid","uid","an","ad","sn","sd", "a", "s"]],  
                 how='left', left_on=["qid","uid"], right_on = ["qid","uid"])
train = pd.merge(train, train_ctr[["qid","uid","CTR"]], how='left', left_on=["qid","uid"], right_on=["qid","uid"])
train = pd.merge(train, train_meanctr[["uid","mctr0","mctr1","mctrgeom"]], how='left', left_on = "uid", right_on ="uid")
train = pd.merge(train, 
                 train_qdp[
                             [
                               "qid","uid", "sh_mean", "sh_cnt", "sh_max", 
                               "sh_min", "cl_mean", "cl_cnt", "cl_max", "cl_min"
                             ]
                         ], how='left', left_on = ["qid","uid"], right_on =["qid","uid"]
                ) 


#%%
test = sample.copy()
test = pd.merge(test, test_sdbn[["qid","uid","an","ad","sn","sd", "a", "s"]],  
                 how='left', left_on=["qid","uid"], right_on = ["qid","uid"])
test = pd.merge(test, test_ctr[["qid","uid","CTR"]], how='left', left_on=["qid","uid"], right_on=["qid","uid"])
test = pd.merge(test, test_meanctr[["uid","mctr0","mctr1","mctrgeom"]], how='left', left_on = "uid", right_on ="uid")
test = pd.merge(test, 
                 test_qdp[
                             [
                               "qid","uid", "sh_mean", "sh_cnt", "sh_max", 
                               "sh_min", "cl_mean", "cl_cnt", "cl_max", "cl_min"
                             ]
                         ], how='left', left_on = ["qid","uid"], right_on =["qid","uid"]
                ) 


#%%
curr_valnames = ["an","ad","sn","sd", "a", "s", "CTR", "mctr0", "mctr1","mctrgeom", "qid","uid", "sh_mean", "sh_cnt", "sh_max", 
                               "sh_min", "cl_mean", "cl_cnt", "cl_max", "cl_min"]
def filter_valid(df):
    return np.isnan(df[curr_valnames].values).sum(1) < len(curr_valnames)


#%%
valid_train = train.iloc[filter_valid(train),:]
X_train = valid_train[curr_valnames].values
y_train = valid_train["mark"].values
valid_train.head()


#%%
valid_test = test.iloc[filter_valid(test), :]
X_test = valid_test[curr_valnames].values
valid_test.head()


#%%
trainqids = set(valid_train.qid.tolist())
np.random.seed(42)
qid_train = np.random.choice(list(trainqids), int(0.8*len(trainqids))).tolist()
qid_test = list(trainqids - set(qid_train))


#%%
pw_train = valid_train[valid_train.qid.apply(lambda x: x in qid_train)]
pw_test = valid_train[valid_train.qid.apply(lambda x: x in qid_test)]
group_train = [a for a in np.bincount(pw_train.qid.tolist()) if a>0]
group_test = [a for a in np.bincount(pw_test.qid.tolist()) if a>0]


#%%
pwX_train = pw_train[curr_valnames].values
pwy_train = pw_train["mark"].values
pwX_test = pw_test[curr_valnames].values
pwy_test = pw_test["mark"].values


#%%
pwy_test[:200]


#%%
pwparams = {
    # Parameters that we are going to tune.
    'max_depth':5,
    'min_child_weight': 1,
    'eta':.1,
    'subsample': 0.9,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'rank:pairwise',
    'eval_metric':'ndcg@5',
    'reg_alpha' : 0.3,
    'reg_lambda': 0.3
}


#%%
import xgboost
pwdTrain = xgboost.DMatrix(pwX_train, label=pwy_train)
pwdTrain.set_group(group_train)
pwdTest = xgboost.DMatrix(pwX_test, label=pwy_test)
pwdTest.set_group(group_test)
pwxgb = xgboost.train(pwparams, pwdTrain, evals = [(pwdTrain, "train"),(pwdTest, "test")], verbose_eval = 1, num_boost_round=10)


#%%
X_train.shape


#%%
fulltrain = xgboost.DMatrix(X_train, label=y_train)
group_train = [a for a in np.bincount(valid_train.qid.tolist()) if a>0]
fulltrain.set_group(group_train)
pwxgb = xgboost.train(pwparams, fulltrain, evals = [(pwdTrain, "train"),(pwdTest, "test")], verbose_eval = 1, num_boost_round=500)


#%%
dSample = xgboost.DMatrix(X_test)
valid_test.loc[:, "mark_pw"] = pwxgb.predict(dSample)
fin_result = valid_test.sort_values(by=['qid',"mark_pw"], ascending = [True, False])


#%%
import datetime
with open('submission_pairwise' + str(datetime.datetime.now()).replace(":","_"), 'w') as out:
    out.write("QueryId,DocumentId\n")
    for qid in set(fin_result.qid.tolist()):
        curr_outs = fin_result[fin_result.qid==qid].uid.tolist()
        counter = 0
        pushed_uids = set()
        for uid in curr_outs:
            counter += 1
            pushed_uids.add(uid)
            out.write("{},{}\n".format(qid,uid))
            if counter == 5: 
                break
        if counter < 5:
            available_outs = set(test[test.qid==0].uid.tolist()) - pushed_uids
            for uid in available_outs:
                counter += 1
                out.write("{},{}\n".format(qid,uid))
                if counter == 5:
                    break


#%%
pw_train.head()