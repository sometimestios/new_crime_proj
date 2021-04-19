import json
import sys
sys.path.append("/home/songhan/miniconda3/envs/py36_for_CrimeClassify/lib/python3.6/site-packages")
reverse_dict = {'婚姻家庭': '0', '劳动纠纷': '1', '交通事故': '2', '债权债务': '3', '刑事辩护': '4', '合同纠纷': '5', '房产纠纷': '6', '侵权': '7',
              '公司法': '8', '医疗纠纷': '9', '拆迁安置': '10', '行政诉讼': '11','建设工程': '12'}
reverse_dict = {'婚姻家庭': '0', '劳动纠纷': '1', '交通事故': '2', '债权债务': '3', '刑事辩护': '4', '合同纠纷': '5', '房产纠纷': '6', '侵权': '7',
              '公司法': '8', '医疗纠纷': '9'}

reverse_dict = {'婚姻家庭': '0', '劳动纠纷': '1', '交通事故': '2', '债权债务': '3', '刑事辩护': '4', '合同纠纷': '5'}
count_dict = {'婚姻家庭': 0, '劳动纠纷': 0, '交通事故': 0, '债权债务': 0, '刑事辩护': 0, '合同纠纷': 0, '房产纠纷': 0, '侵权': 0,
              '公司法': 0, '医疗纠纷': 0, '拆迁安置': 0, '行政诉讼': 0, '建设工程': 0}

def deal(src_path, des_path):
    src_file = open(src_path, 'r', encoding='utf-8')
    des_file = open(des_path, 'w', encoding='utf-8')
    count = 0
    crime_set = set([])
    for line in src_file.readlines():
        fileJson = json.loads(line)
        question = repr(fileJson["question"]).strip('\'')
        cate=fileJson["category"]
        if cate not in reverse_dict:
            continue
        if count_dict[cate]>5000-1:
            continue
        lable = reverse_dict[cate]
        count_dict[cate]+=1
        des_file.write(question + "##" + lable + "\n")
        count += 1
        if count >= 30000:
            break
    print("{} case counted".format(count))
    print(count_dict)
    src_file.close()
    des_file.close()

deal("data/qa_corpus.json", "data/6fenlei3W_question_train.txt")

