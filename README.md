源自https://github.com/liuhuanyong/CrimeKgAssitant
# CrimeKgAssitant
Crime assistant including crime type prediction and crime consult service based on nlp methods and crime kg,罪名法务智能项目,内容包括856项罪名知识图谱, 基于280万罪名训练库的罪名预测,基于20W法务问答对的13类问题分类与法律资讯问答功能.

# 项目功能
目前知识图谱在各个行业中应用逐步打开,尤其在金融,医疗,法律,旅游方面.知识图谱助力法律智能,能够在一定程度上利用现有大数据以及机器学习/深度学习与自然语言处理技术,提供一些智能的解决方案.本项目将完成两个大方向的工作:
1, 以罪名为核心,收集相关数据,建成基本的罪名知识图谱,法务资讯对话知识库,案由量刑知识库.
2, 分别基于步骤1的结果,完成以下四个方面的工作:
1) 基于案由量刑知识库的罪名预测模型
2) 基于法务咨询对话知识库的法务问题类型分类
3) 基于法务咨询对话知识库的法务问题自动问答服务
4) 基于罪行知识图谱的知识查询

# 罪名预测
1, 问题类型:
罪名一共包括202种罪名,文件放在dict/crime.txt中, 详细内容举例如下:

        妨害公务
        寻衅滋事
        盗窃、侮辱尸体
        危险物品肇事
        非法采矿
        组织、强迫、引诱、容留、介绍卖淫
        开设赌场
        聚众斗殴
        绑架
        非法持有毒品
        销售假冒注册商标的商品
        容留他人吸毒
        假冒注册商标
        交通肇事
        破坏电力设备
        组织卖淫
        合同诈骗
        走私武器、弹药
        抢劫
        非法处置查封、扣押、冻结的财产

2, 问题模型:
罪刑数据库一共有288万条训练数据,要做的是202类型的罪名多分类问题.本项目采用的方式为:

| 训练数据规模 | 数据向量表示 | 模型 |训练时长 | 准确率 |
| :--- | :---: | :---: | :--- | :--- |
| 20W | doc embedding | svm | 0.5h| 0.83352184|
| 288W | doc embedding | svm | 12h| 0.9203119|

3, 效果:
执行 python crime_classify.py

    crime desc:这宗案情凶残的案件中，受害人樊敏仪是一名夜总会舞女，1997年因筹措祖母的医药费，偷取任职皮条客的首被告陈文乐数千元港币及其他财物(另一说是指毒品债)。首被告陈文乐于是吩咐次被告梁胜祖及第三被告梁伟伦向女受害人追债。女受害人为求还清债项，怀孕后仍继续接客，3名被告将欠款不断提高，受害人因无力偿还，因而触怒三人。1999年3月17日梁胜祖及梁伟伦按照首被告要求，将受害人从葵涌丽瑶邨富瑶楼一单位押走，禁锢于尖沙咀加连威老道31号3楼一单位。当回到单位后，梁伟伦质问受害人为何不还钱、为何不肯回电话，连踢受害人超过50次。3名被告用木板封着该单位的玻璃窗，以滚油泼向受害人的口腔，在伤口上涂上辣椒油，逼她吞吃粪便及喝尿。被告之后把烧溶的塑胶吸管滴在她的腿上，并命令受害人发出笑声。受害人开始神志不清，并不时挑起伤口上的焦疤，被告于是以电线紧紧捆缠受害人双手多个小时，之后又用铁棍殴打她双手。
    crime label: 非法拘禁
    *********************************************************
    crime desc:有很多人相信是莉齐进行了这次谋杀，虽然她始终没有承认，陪审团也得出了她无罪的结论。莉齐·鲍顿是一个32岁的老姑娘，她被指控用刀杀死了自己的父亲和继母。虽然她最后无罪获释，但人们知道，她对继母一直怀恨在心，而在谋杀发生的前一天，她曾预言了将要发生的事。凶杀案发生时她已30岁。1892年8月4日中午，莉齐·鲍顿叫唤她的邻居说，她的父亲被杀了，警察到来时，发现她的母亲也死了。母亲被斧子砍了18下，父亲被砍了10下。消息立即被传开了，媒体认为莉齐本人极有谋杀嫌疑。然而次年六月，法庭宣判莉齐无罪。此后，她的故事广为流传，被写成了小说，芭蕾，百老汇，歌剧。最后是日本的教科书将她的童谣作为鹅妈妈童话收录的。
    crime label: 故意杀人
    *********************************************************
    crime desc:017年5月26日11时许，被告人陈某、李某林与一同前去的王某，在信阳市羊山新区中级人民法院工地南大门门口，拦住被害人张某军，对其进行殴打，致其右手受伤，损伤程度属轻伤一级。2017年7月22日，李某林主动到信阳市公安局羊山分局投案。在审理过程中，被告人陈某、李某林与被害人张某军自愿达成赔偿协议，由陈某、李某林赔偿张祖军全部经济损失共计10万元，张某军对二被告人予以谅解。
    crime label: 故意伤害
    *********************************************************
    crime desc:被告人赵某某于1999年5月起在某医院眼科开展医师执业活动，2010年11月其与医院签订事业单位聘用合同，从事专业技术工作，并于2011年取得临床医学主任医师职称。2014年3月起其担任眼科主任，在院长、分管院长和医务科领导下负责本科医疗、教学、科研和行政管理等工作。赵某某担任眼科主任期间，利用职务之便，收受人工晶体供货商给付的回扣共计37万元。赵某某作为眼科主任，在医院向供货商订购进口人工晶体过程中，参与了询价、谈判、合同签订和采购的过程。2015年4月12日，赵某某接受检察院调查，如实供述了收受人工晶体销售商回扣的事实。
    crime label: 受贿
    *********************************************************
    crime desc:金陵晚报报道 到人家家里偷东西，却没发现可偷之物，丧尽天良的小偷为了报复竟将屋内熟睡的老太太强奸。日前，卢勇(化名) 在潜逃了一年后因再次出手被抓获。　　 31岁的卢勇是安徽枞阳县人，因家境贫寒，到现在仍是单身。今年6月份，他从老家来到南京，连续作案多起。7月1日凌晨，当他窜至莫愁新村再次作案时，当场被房主抓获。　　经审讯又查明，去年8月30日清晨4时许，卢勇来宁行窃未遂后，贼心不死。又到附近的另一户人家行窃。他在房内找了一圈都没找到任何值钱的东西，只有个女人在床上睡觉。卢勇觉得没偷到东西亏了，想报复一下这户人家，就走到床边捂住女人的嘴，不顾反抗将其强奸后逃跑。　　据卢勇供述，他当时并没注意女人的年纪，直到事后他才发现对方竟然是个早已上了年纪的老太太。日前，卢勇因涉嫌盗窃和强奸被检方审查起诉。
    crime label: 强奸

# 法务咨询问题分类
1, 问题类型:
法务资讯问题一共包括13类,详细内容如下:

        0: "婚姻家庭",
        1: "劳动纠纷",
        2: "交通事故",
        3: "债权债务",
        4: "刑事辩护",
        5: "合同纠纷",
        6: "房产纠纷",
        7: "侵权",
        8: "公司法",
        9: "医疗纠纷",
        10: "拆迁安置",
        11: "行政诉讼",
        12: "建设工程"
2, 问题模型:
法务咨询数据库一共有20万条训练数据,要做的是13类型咨询问题多分类问题.本项目采用的方式为:

| 训练数据规模 |测试集规模 | 模型 |训练时长 | 训练集准确率 |测试集准确率|
| :--- | :---: | :---: | :--- | :--- | :--- |
| 4W | 1W | CNN | 15*20s| 0.984|0.959|
| 4W | 1W | LSTM | 51*20s| 0.838|0.717|

3, 效果:
执行 python question_classify.py

    question desc:他们俩夫妻不和睦,老公总是家暴,怎么办
    question_type: 婚姻家庭 0.9994359612464905
    *********************************************************
    question desc:我们老板总是拖欠工资怎么办,怎么起诉他
    question_type: 劳动纠纷 0.9999903440475464
    *********************************************************
    question desc:最近p2p暴雷,投进去的钱全没了,能找回来吗
    question_type: 刑事辩护 0.3614000678062439
    *********************************************************
    question desc:有人上高速,把车给刮的不像样子,如何是好
    question_type: 交通事故 0.9999163150787354
    *********************************************************
    question desc:有个老头去世了,儿女们在争夺财产,闹得不亦乐乎
    question_type: 婚姻家庭 0.9993444085121155

# 法务咨询自动问答
运行 python crime_qa.py

    question:朋友欠钱不还咋办
    answers: ['欠款金额是多少 ', '多少钱呢', '律师费诉讼费都非常少都很合理，一定要起诉。', '大概金额多少？', '需要看标的额和案情复杂程度，建议细致面谈']
    *******************************************************
    question:昨天把人家车刮了,要赔多少
    answers: ['您好，建议协商处理，如果对方告了你们，就只能积极应诉了。', '您好，建议尽量协商处理，协商不成可起诉']
    *******************************************************
    question:最近丈夫经常家暴,我受不了了
    answers: ['报警要求追究刑事责任。', '您好，建议起诉离婚并请求补偿。', '你好！可以起诉离婚，并主张精神损害赔偿。']
    *******************************************************
    question:毕业生拿了户口就跑路可以吗
    answers: 您好,对于此类问题,您可以咨询公安部门
    *******************************************************
    question:孩子离家出走,怎么找回来
    answers: ['孩子父母没有结婚，孩子母亲把孩子带走了？这样的话可以起诉要求抚养权的。毕竟母亲也是孩子的合法监护人，报警警察一般不受理。']
    *******************************************************
    question:村霸把我田地给占了,我要怎么起诉
    answers: ['可以向上级主管部门投诉解决', '您好，您可以及时向土地管理部门投诉的！', '对方侵权，可以向法院起诉。', '你好，对方侵权，可以向法院起诉。', '你好，可起诉处理，一、当事人起诉，首先应提交起诉书，并按对方当事人人数提交相应份数的副本。当事人是公民的，应写明双方当事人的姓名、性别、年龄、籍贯、住址;当事人是单位的，应写明单位名称、地址、法定代表人或负责人姓名。起诉书正文应写明请求事项和起诉事实、理由，尾部须署名或盖公章。二、根据"谁主张谁举证"原则，原告向法院起诉应提交下列材料：1、原告主体资格的材料。如居民身份证、户口本、护照、港澳同胞回乡证、结婚证等证据的原件和复印件;企业单位作为原告的应提交营业执照、商业登记证明等材料的复印件。2、证明原告诉讼主张的证据。如合同、协议、债权文书(借条、欠条等)、收发货凭证、往来信函等。', '您好，起诉维权。', '您好，可以起诉解决。']
    *******************************************************
    question:售卖危违禁物品,有什么风险
    answers: ['没什么']
    *******************************************************
    question:找不到女朋友啊..
    answers: 您好,对于此类问题,您可以咨询公安部门
    *******************************************************
    question:我要离婚
    answers: ['现在就可向法院起诉离婚。', '不需要分开两年起诉离婚。感情完全破裂就可以提起诉讼离婚。', '你可以直接起诉离婚', '直接起诉']
    *******************************************************
    question:醉驾,要坐牢吗
    answers: ['要负刑事责任很可能坐牢', '由警方处理.,']
    *******************************************************
    question:你好，我向大学提出退学申请，大学拒绝，理由是家长不同意。我该怎么办？
    answers: ['自己可决定的 ']
    *******************************************************
    question:请问在上班途中，出车祸我的责任偏大属于工伤吗？
    answers: ['属于工伤']
    *******************************************************
    question:结婚时女方拿了彩礼就逃了能要回来吗
    answers: ['可以要求退还彩礼。，']
    *******************************************************
    question:房产证上是不是一定要写夫妻双方姓名
    answers: ['可以不填，即使一个人的名字，婚后买房是共同财产。', '不是必须的', '可以写一方名字，对方公证，证明该房产系你一人财产', '你好，不是必须']
    *******************************************************
    question:儿女不履行赡养义务是不是要判刑
    answers: ['什么情况了？']
    *******************************************************
    question:和未成年人发生关系,需要坐牢吗
    answers: ['女孩子在发生关系的时候是否满14周岁，如果是且自愿就不是犯罪', '你好，如果是双方愿意的情况下是不犯法的。', '发生性关系时已满十四岁并且是自愿的依法律规定不构成强奸罪，不构成犯罪的。', '若是自愿，那就没什么可说了。', '双方愿意不犯法', '你好 如果是自愿的 不犯法 ', '自愿的就没事']
    *******************************************************
    question:撞死人逃跑要怎么处理
    answers: ['等待警察处理。，']

# 总结
1, 本项目实现的是以罪刑为核心的法务应用落地的一个demo尝试.  
2, 本项目采用机器学习,深度学习的方法完成了罪名预测,客服问句类型预测多分类任务,取得了较好的性能,模型可以直接使用.  
3, 本项目构建起了一个20万问答集,856个罪名的知识库,分别存放在data/kg_crime.json和data/qa_corpus.json文件中.  
4, 法务问答,可以是智能客服在法律资讯网站中的一个应用场景落地. 本项目采用的是ES+语义相似度加权打分策略实现的问答技术路线, 权值计算与阈值设定可以用户指定.  
5, 对于罪名知识图谱中的知识可以进一步进行结构化处理,这是后期可以完善的地方.  
6, 如何将罪名,咨询,智能研判结合在一起,形成通路,其实可以进一步提升知识图谱在法务领域的应用.  

# contact 
如有自然语言处理、知识图谱、事理图谱、社会计算、语言资源建设等问题或合作，请联系我:  
邮箱:lhy_in_blcu@126.com  
csdn:https://blog.csdn.net/lhy2014  
我的自然语言处理项目: https://liuhuanyong.github.io/  
刘焕勇，中国科学院软件研究所  
