# 保证标签集合原始一样。有问题的全部替换成原始语句
import os
import json
from collections import Counter
ACT = [
    "Inform",
    "AskForSure",
    "Deny",
    "Explanation",
    "Assure",
    "AskFor",
    "Advice",
    "GeneralAdvice",
    "Chitchat",
    "AdviceNot",
    "AskHow",
    "AskWhy",
    "Uncertain",
    "Accept",
    "GeneralExplanation"
]
DOMAIN = [
    "饮食",
    "问题",
    "行为",
    "",
    "治疗",
    "运动",
    "基本信息"
]
SLOT = [
    "饮食名",
    "血糖值",
    "行为名",
    "",
    "饮食量",
    "药品",
    "持续时长",
    "时间",
    "检查项",
    "检查值",
    "用药（治疗）频率",
    "用药量",
    "运动名",
    "疾病",
    "症状",
    "成分",
    "成份量",
    "频率",
    "效果",
    "治疗名",
    "部位",
    "体重",
    "身高",
    "症状部位",
    "状态",
    "年龄",
    "强度",
    "药品类型",
    "适应症",
    "既往史",
    "性别"
]

INTENT_VOCAB=[
  "Deny+++",
  "Assure+++",
  "AskFor+问题+血糖值+?",
  "AskFor+治疗+药品+?",
  "GeneralAdvice+饮食++",
  "GeneralAdvice+运动++",
  "Chitchat+++",
  "GeneralAdvice+治疗++",
  "AskHow+++",
  "AskFor+治疗+检查值+?",
  "AskWhy+++",
  "AskFor+问题+持续时长+?",
  "Uncertain+++",
  "Accept+++",
  "GeneralExplanation+饮食++",
  "GeneralExplanation+行为++",
  "GeneralExplanation+运动++",
  "GeneralExplanation+治疗++",
  "AskFor+饮食+饮食量+?",
  "AskFor+问题+疾病+?",
  "AskFor+基本信息+体重+?",
  "GeneralExplanation+问题++",
  "AskFor+基本信息+年龄+?",
  "AskFor+治疗+检查项+?",
  "AskFor+行为+效果+?",
  "AskFor+饮食+饮食名+?",
  "AskFor+问题+症状+?",
  "AskFor+行为+行为名+?",
  "AskFor+运动+强度+?",
  "AskFor+行为+频率+?",
  "AskFor+运动+运动名+?",
  "AskFor+问题+时间+?",
  "AskFor+治疗+效果+?",
  "AskFor+治疗+治疗名+?",
  "AskFor+基本信息+既往史+?",
  "GeneralExplanation+基本信息++",
  "AskFor+问题+症状部位+?",
  "GeneralExplanation+++",
  "AskFor+治疗+持续时长+?",
  "GeneralAdvice+行为++",
  "AskFor+治疗+用药（治疗）频率+?",
  "AskFor+治疗+时间+?",
  "AskFor+治疗+用药量+?",
  "AskFor+行为+时间+?",
  "AskFor+基本信息+身高+?",
  "GeneralAdvice+++",
  "AskFor+行为+持续时长+?",
  "AskForSure+基本信息+体重+?",
  "GeneralAdvice+问题++",
  "AskFor+问题+状态+?",
  "AskFor+治疗+部位+?",
  "AskFor+饮食+时间+?",
  "AskFor+基本信息+性别+?",
  "AskHow+行为+效果+?",
  "AskFor+运动+时间+?",
  "AskFor+运动+持续时长+?",
  "AskFor+治疗+药品类型+?"
]
TAG_VOCAB=[
  "O",
  "B+Inform+饮食+饮食名",
  "I+Inform+饮食+饮食名",
  "B+AskForSure+行为+行为名",
  "B+Inform+问题+血糖值",
  "I+Inform+问题+血糖值",
  "B+Explanation+饮食+饮食名",
  "B+Explanation+饮食+饮食量",
  "I+AskForSure+行为+行为名",
  "B+Inform+问题+疾病",
  "I+Inform+问题+疾病",
  "B+Inform+问题+时间",
  "I+Inform+问题+时间",
  "B+AskFor+问题+时间",
  "I+AskFor+问题+时间",
  "B+Inform+行为+行为名",
  "I+Inform+行为+行为名",
  "B+Explanation+问题+疾病",
  "I+Explanation+问题+疾病",
  "B+Inform+治疗+药品",
  "I+Inform+治疗+药品",
  "B+Inform+治疗+用药（治疗）频率",
  "I+Inform+治疗+用药（治疗）频率",
  "B+Inform+治疗+用药量",
  "I+Inform+治疗+用药量",
  "B+Advice+行为+行为名",
  "I+Advice+行为+行为名",
  "B+Inform+行为+持续时长",
  "I+Inform+行为+持续时长",
  "B+AskForSure+问题+时间",
  "I+AskForSure+问题+时间",
  "B+AskForSure+问题+血糖值",
  "B+Inform+行为+时间",
  "I+Inform+行为+时间",
  "B+Explanation+问题+时间",
  "I+Explanation+问题+时间",
  "B+Explanation+问题+血糖值",
  "I+Explanation+问题+血糖值",
  "B+Inform+运动+运动名",
  "I+Inform+运动+运动名",
  "B+Advice+运动+运动名",
  "I+Advice+运动+运动名",
  "B+Inform+治疗+检查项",
  "I+Inform+治疗+检查项",
  "B+Inform+治疗+检查值",
  "I+Inform+治疗+检查值",
  "B+Inform+治疗+持续时长",
  "I+Inform+治疗+持续时长",
  "B+AskForSure+饮食+饮食名",
  "I+AskForSure+饮食+饮食名",
  "B+AskForSure+饮食+饮食量",
  "I+AskForSure+饮食+饮食量",
  "B+Inform+治疗+时间",
  "I+Inform+治疗+时间",
  "B+AskForSure+运动+时间",
  "I+AskForSure+运动+时间",
  "B+AskForSure+运动+运动名",
  "I+AskForSure+运动+运动名",
  "B+Inform+饮食+时间",
  "I+Inform+饮食+时间",
  "B+Inform+饮食+饮食量",
  "I+Inform+饮食+饮食量",
  "B+AdviceNot+问题+疾病",
  "I+AdviceNot+问题+疾病",
  "B+AdviceNot+行为+行为名",
  "I+AdviceNot+行为+行为名",
  "B+Inform+问题+症状",
  "I+Inform+问题+症状",
  "B+Explanation+问题+症状",
  "I+Explanation+问题+症状",
  "B+Advice+饮食+饮食量",
  "I+Advice+饮食+饮食量",
  "B+Advice+饮食+饮食名",
  "I+Advice+饮食+饮食名",
  "B+Explanation+饮食+成分",
  "B+Explanation+饮食+成份量",
  "I+Explanation+饮食+成份量",
  "B+AskForSure+治疗+用药量",
  "B+Advice+行为+时间",
  "I+Advice+行为+时间",
  "B+Explanation+问题+持续时长",
  "I+Explanation+问题+持续时长",
  "B+AskForSure+问题+疾病",
  "I+AskForSure+问题+疾病",
  "B+AskHow+问题+症状",
  "I+AskHow+问题+症状",
  "B+Advice+问题+疾病",
  "I+Advice+问题+疾病",
  "B+Inform+问题+症状部位",
  "I+Inform+问题+症状部位",
  "B+AskFor+问题+症状",
  "I+AskFor+问题+症状",
  "B+Inform+问题+持续时长",
  "I+Inform+问题+持续时长",
  "B+Advice+治疗+检查项",
  "I+Advice+治疗+检查项",
  "B+AskForSure+问题+症状",
  "I+AskForSure+问题+症状",
  "B+Inform+运动+持续时长",
  "I+Inform+运动+持续时长",
  "B+AskForSure+治疗+检查项",
  "I+AskForSure+治疗+检查项",
  "B+Inform+治疗+部位",
  "I+Inform+治疗+部位",
  "B+AskFor+行为+行为名",
  "I+AskFor+行为+行为名",
  "B+Inform+行为+频率",
  "I+Inform+行为+频率",
  "B+Advice+饮食+时间",
  "I+Advice+饮食+时间",
  "B+AskHow+行为+行为名",
  "I+AskHow+行为+行为名",
  "B+AskFor+饮食+时间",
  "I+AskFor+饮食+时间",
  "B+Inform+行为+效果",
  "I+Inform+行为+效果",
  "B+Explanation+行为+行为名",
  "I+Explanation+行为+行为名",
  "B+AskForSure+治疗+治疗名",
  "I+AskForSure+治疗+治疗名",
  "B+AskForSure+治疗+部位",
  "I+AskForSure+治疗+部位",
  "B+Explanation+治疗+治疗名",
  "I+Explanation+治疗+治疗名",
  "B+Explanation+治疗+部位",
  "I+Explanation+治疗+部位",
  "B+Explanation+治疗+效果",
  "I+Explanation+治疗+效果",
  "B+AdviceNot+问题+时间",
  "I+AdviceNot+问题+时间",
  "B+AdviceNot+治疗+部位",
  "I+AdviceNot+治疗+部位",
  "B+Advice+治疗+部位",
  "I+Advice+治疗+部位",
  "B+Inform+基本信息+体重",
  "I+Inform+基本信息+体重",
  "B+Inform+基本信息+身高",
  "I+Inform+基本信息+身高",
  "B+Explanation+治疗+时间",
  "I+Explanation+治疗+时间",
  "B+Explanation+治疗+检查项",
  "I+Explanation+治疗+检查项",
  "B+Advice+行为+效果",
  "I+Advice+行为+效果",
  "B+AskFor+问题+疾病",
  "I+AskFor+问题+疾病",
  "B+Inform+治疗+治疗名",
  "I+Inform+治疗+治疗名",
  "I+AskForSure+问题+血糖值",
  "B+AskForSure+问题+状态",
  "I+AskForSure+问题+状态",
  "B+AskForSure+治疗+药品",
  "I+AskForSure+治疗+药品",
  "B+Advice+问题+时间",
  "I+Advice+问题+时间",
  "B+Advice+问题+血糖值",
  "I+Advice+问题+血糖值",
  "B+Inform+基本信息+年龄",
  "B+Advice+治疗+药品",
  "I+Advice+治疗+药品",
  "B+AskFor+治疗+时间",
  "I+AskFor+治疗+时间",
  "B+AskFor+治疗+治疗名",
  "I+AskFor+治疗+治疗名",
  "B+AskForSure+治疗+时间",
  "I+AskForSure+治疗+时间",
  "B+AskForSure+治疗+检查值",
  "I+AskForSure+治疗+检查值",
  "B+Explanation+治疗+检查值",
  "B+AskForSure+行为+频率",
  "I+AskForSure+行为+频率",
  "B+Advice+运动+强度",
  "I+Advice+运动+强度",
  "B+AskFor+饮食+饮食量",
  "I+AskFor+饮食+饮食量",
  "B+AskFor+饮食+饮食名",
  "I+AskFor+饮食+饮食名",
  "I+Explanation+饮食+饮食名",
  "B+AskForSure+治疗+效果",
  "I+AskForSure+治疗+效果",
  "B+AskForSure+治疗+药品类型",
  "I+AskForSure+治疗+药品类型",
  "B+Explanation+治疗+药品类型",
  "I+Explanation+治疗+药品类型",
  "B+Advice+治疗+时间",
  "I+Advice+治疗+时间",
  "B+AskHow+问题+疾病",
  "I+AskHow+问题+疾病",
  "B+Advice+行为+持续时长",
  "I+Advice+行为+持续时长",
  "B+AskFor+行为+效果",
  "I+AskFor+行为+效果",
  "B+Inform+运动+时间",
  "I+Inform+运动+时间",
  "B+Explanation+运动+强度",
  "I+Explanation+运动+强度",
  "B+AskForSure+行为+时间",
  "I+AskForSure+行为+时间",
  "B+Advice+行为+频率",
  "I+Advice+行为+频率",
  "B+Inform+问题+状态",
  "I+Inform+问题+状态",
  "B+Inform+治疗+药品类型",
  "I+Inform+治疗+药品类型",
  "B+Advice+治疗+药品类型",
  "I+Advice+治疗+药品类型",
  "B+AskHow+运动+运动名",
  "I+AskHow+运动+运动名",
  "B+AskForSure+运动+强度",
  "B+Advice+运动+时间",
  "I+Advice+运动+时间",
  "B+Advice+治疗+治疗名",
  "I+Advice+治疗+治疗名",
  "B+Explanation+治疗+药品",
  "I+Explanation+治疗+药品",
  "B+AskHow+治疗+时间",
  "I+AskHow+治疗+时间",
  "B+AskHow+治疗+检查项",
  "I+AskHow+治疗+检查项",
  "B+AskFor+治疗+药品类型",
  "I+AskFor+治疗+药品类型",
  "B+Advice+治疗+用药（治疗）频率",
  "I+Advice+治疗+用药（治疗）频率",
  "B+Advice+饮食+成分",
  "B+Advice+饮食+成份量",
  "I+Advice+饮食+成份量",
  "I+Advice+饮食+成分",
  "B+AskFor+治疗+药品",
  "I+AskFor+治疗+药品",
  "B+AskHow+治疗+药品",
  "B+Explanation+行为+效果",
  "I+Explanation+行为+效果",
  "B+AskHow+问题+症状部位",
  "I+AskHow+问题+症状部位",
  "B+Advice+问题+症状",
  "I+Advice+问题+症状",
  "B+AskFor+问题+症状部位",
  "I+AskFor+问题+症状部位",
  "B+Advice+治疗+用药量",
  "I+Advice+治疗+用药量",
  "B+AskFor+问题+血糖值",
  "I+AskFor+问题+血糖值",
  "B+Inform+饮食+成份量",
  "I+Inform+饮食+成份量",
  "B+Inform+饮食+成分",
  "B+Inform+治疗+效果",
  "I+Inform+治疗+效果",
  "B+AskForSure+行为+效果",
  "I+AskForSure+行为+效果",
  "B+AskFor+治疗+用药量",
  "I+AskFor+治疗+用药量",
  "I+AskHow+治疗+药品",
  "B+AskForSure+基本信息+体重",
  "I+AskForSure+基本信息+体重",
  "B+Explanation+基本信息+体重",
  "I+Explanation+基本信息+体重",
  "B+AskFor+治疗+检查项",
  "I+AskFor+治疗+检查项",
  "I+Inform+基本信息+年龄",
  "B+Explanation+行为+时间",
  "I+Explanation+行为+时间",
  "B+Explanation+治疗+适应症",
  "I+Explanation+治疗+适应症",
  "B+AskHow+问题+血糖值",
  "I+AskHow+问题+血糖值",
  "B+Explanation+问题+症状部位",
  "I+Explanation+问题+症状部位",
  "I+AskForSure+运动+强度",
  "I+Explanation+治疗+检查值",
  "B+AskHow+治疗+治疗名",
  "I+AskHow+治疗+治疗名",
  "B+AskForSure+问题+症状部位",
  "B+AskFor+治疗+适应症",
  "I+AskFor+治疗+适应症",
  "B+Advice+治疗+效果",
  "I+Advice+治疗+效果",
  "B+AskHow+问题+状态",
  "I+AskHow+问题+状态",
  "I+AskForSure+问题+症状部位",
  "B+Explanation+问题+状态",
  "I+Explanation+问题+状态",
  "B+Inform+基本信息+既往史",
  "I+Inform+基本信息+既往史",
  "B+Inform+运动+强度",
  "I+Inform+运动+强度",
  "B+Advice+治疗+持续时长",
  "I+Advice+治疗+持续时长",
  "B+AskHow+治疗+检查值",
  "I+AskHow+治疗+检查值",
  "B+Inform+运动+频率",
  "I+Inform+运动+频率",
  "B+Explanation+饮食+时间",
  "I+Explanation+饮食+时间",
  "I+Explanation+饮食+成分",
  "B+Explanation+运动+运动名",
  "I+Explanation+运动+运动名",
  "B+AdviceNot+治疗+药品类型",
  "I+AdviceNot+治疗+药品类型",
  "B+Advice+运动+持续时长",
  "I+Advice+运动+持续时长",
  "B+AskFor+基本信息+既往史",
  "I+AskFor+基本信息+既往史",
  "B+AskFor+治疗+检查值",
  "I+AskFor+治疗+检查值",
  "I+AskForSure+治疗+用药量",
  "B+AskForSure+治疗+行为名",
  "I+AskForSure+治疗+行为名",
  "B+AskForSure+基本信息+身高",
  "I+AskForSure+基本信息+身高",
  "B+Explanation+治疗+用药（治疗）频率",
  "I+Explanation+治疗+用药（治疗）频率",
  "I+Inform+饮食+成分",
  "B+Inform+治疗+适应症",
  "I+Inform+治疗+适应症",
  "B+AskForSure+饮食+时间",
  "I+AskForSure+饮食+时间",
  "B+Advice+基本信息+体重",
  "I+Advice+基本信息+体重",
  "B+AskForSure+基本信息+年龄",
  "B+AdviceNot+饮食+成分",
  "I+AdviceNot+饮食+成分",
  "B+AdviceNot+饮食+饮食名",
  "I+AdviceNot+饮食+饮食名",
  "B+AdviceNot+饮食+饮食量",
  "B+AskFor+饮食+成份量",
  "I+AskFor+饮食+成份量",
  "B+AdviceNot+问题+血糖值",
  "B+AskFor+行为+频率",
  "I+AskFor+行为+频率",
  "I+AskForSure+基本信息+年龄",
  "B+AskHow+行为+时间",
  "I+AskHow+行为+时间",
  "B+Explanation+行为+持续时长",
  "I+Explanation+行为+持续时长",
  "B+AdviceNot+治疗+药品",
  "I+AdviceNot+治疗+药品",
  "B+AskForSure+饮食+成分",
  "I+AskForSure+饮食+成分",
  "B+AskForSure+问题+持续时长",
  "I+AskForSure+问题+持续时长",
  "B+Inform+运动+效果",
  "I+Inform+运动+效果",
  "B+AdviceNot+行为+频率",
  "I+AdviceNot+行为+频率",
  "B+AskForSure+饮食+效果",
  "B+Explanation+饮食+效果",
  "I+Explanation+饮食+效果",
  "B+AskFor+治疗+效果",
  "I+AskFor+治疗+效果",
  "I+AdviceNot+饮食+饮食量",
  "B+AskHow+治疗+部位",
  "I+AskHow+治疗+部位",
  "B+AskFor+行为+时间",
  "I+AskFor+行为+时间",
  "B+AskFor+运动+时间",
  "I+AskFor+运动+时间",
  "I+Explanation+饮食+饮食量",
  "B+Uncertain+问题+疾病",
  "I+Uncertain+问题+疾病",
  "B+Advice+问题+状态",
  "I+Advice+问题+状态",
  "B+Advice+运动+效果",
  "I+Advice+运动+效果",
  "B+AskHow+运动+时间",
  "B+AdviceNot+饮食+时间",
  "I+AdviceNot+饮食+时间",
  "B+Advice+饮食+效果",
  "I+Advice+饮食+效果",
  "B+Advice+运动+频率",
  "I+Advice+运动+频率",
  "B+AskForSure+治疗+持续时长",
  "I+AskForSure+治疗+持续时长",
  "B+Advice+问题+症状部位",
  "I+Advice+问题+症状部位",
  "B+Explanation+治疗+用药量",
  "I+Explanation+治疗+用药量",
  "B+AskHow+行为+效果",
  "I+AskHow+行为+效果",
  "B+AdviceNot+行为+持续时长",
  "I+AdviceNot+行为+持续时长",
  "B+AdviceNot+问题+症状",
  "I+AdviceNot+问题+症状",
  "B+AskFor+治疗+部位",
  "I+AskFor+治疗+部位",
  "B+Explanation+基本信息+既往史",
  "I+Explanation+基本信息+既往史",
  "B+Explanation+基本信息+年龄",
  "I+Explanation+基本信息+年龄",
  "B+AskHow+饮食+饮食量",
  "B+AdviceNot+运动+强度",
  "I+AdviceNot+运动+强度",
  "I+AskForSure+饮食+效果",
  "B+AskFor+饮食+效果",
  "I+AskFor+饮食+效果",
  "B+AskForSure+基本信息+既往史",
  "I+AskForSure+基本信息+既往史",
  "B+AskHow+问题+时间",
  "I+AskHow+问题+时间",
  "B+AskHow+治疗+用药量",
  "I+AskHow+治疗+用药量",
  "B+Explanation+运动+频率",
  "B+AskForSure+治疗+用药（治疗）频率",
  "I+AskForSure+治疗+用药（治疗）频率",
  "B+AskFor+运动+运动名",
  "I+AskFor+运动+运动名",
  "B+AskHow+治疗+药品类型",
  "I+AskHow+治疗+药品类型",
  "B+AskForSure+运动+频率",
  "I+AskForSure+运动+频率",
  "B+Inform+基本信息+性别",
  "B+AskHow+饮食+饮食名",
  "I+AskHow+饮食+饮食名",
  "B+AdviceNot+饮食+成份量",
  "I+AdviceNot+饮食+成份量",
  "B+Explanation+基本信息+性别",
  "B+Uncertain+行为+行为名",
  "I+Uncertain+行为+行为名",
  "B+Explanation+行为+频率",
  "I+Explanation+行为+频率",
  "B+Advice+治疗+检查值",
  "I+Advice+治疗+检查值",
  "B+AdviceNot+运动+运动名",
  "I+AdviceNot+运动+运动名",
  "I+Inform+基本信息+性别",
  "B+AskForSure+运动+效果",
  "I+AskForSure+运动+效果",
  "B+AdviceNot+运动+时间",
  "I+AdviceNot+运动+时间",
  "B+AskFor+饮食+成分",
  "I+AskFor+饮食+成分",
  "B+Advice+治疗+适应症",
  "I+Advice+治疗+适应症",
  "B+AskHow+基本信息+性别",
  "I+AskHow+基本信息+性别",
  "B+Uncertain+问题+症状",
  "I+Uncertain+问题+症状",
  "B+AskFor+问题+状态",
  "I+AskFor+问题+状态",
  "B+AskForSure+运动+持续时长",
  "I+AskForSure+运动+持续时长",
  "B+AskHow+饮食+效果",
  "B+Explanation+治疗+持续时长",
  "I+Explanation+治疗+持续时长",
  "B+AskForSure+基本信息+性别",
  "B+Explanation+运动+持续时长",
  "I+Explanation+运动+持续时长",
  "B+AskForSure+饮食+成份量",
  "I+AskForSure+饮食+成份量",
  "B+AskHow+运动+持续时长",
  "I+AskHow+运动+持续时长",
  "B+AdviceNot+行为+时间",
  "I+AdviceNot+行为+时间",
  "B+GeneralAdvice+问题+症状",
  "I+GeneralAdvice+问题+症状",
  "B+GeneralAdvice+行为+行为名",
  "B+AskFor+治疗+用药（治疗）频率",
  "I+AskFor+治疗+用药（治疗）频率",
  "B+AskHow+基本信息+体重",
  "I+AskHow+基本信息+体重",
  "I+AskForSure+基本信息+性别",
  "B+AdviceNot+治疗+治疗名",
  "I+AdviceNot+治疗+治疗名",
  "I+AskHow+饮食+饮食量",
  "B+AskHow+问题+持续时长",
  "I+AskHow+问题+持续时长",
  "B+AdviceNot+治疗+检查项",
  "I+AdviceNot+治疗+检查项",
  "B+AdviceNot+治疗+用药量",
  "I+AdviceNot+治疗+用药量",
  "B+AskFor+行为+持续时长",
  "I+AskFor+行为+持续时长",
  "B+AskForSure+治疗+适应症",
  "I+AskForSure+治疗+适应症"
]

target_list = []
# 获取当前文件路径
# /yangjf/students/zhou/project/diachatbot/data/diachat/preprocess_augment.py
file = os.path.abspath(__file__)
# 获取当前文件所在目录
# /yangjf/students/zhou/project/diachatbot/data/diachat
path = os.path.dirname(file)
# print(path)
# 路径拼接
raw_file= os.path.join(path, 'annotations_20220914_2.json')
# /yangjf/students/zhou/project/diachatbot/data/diachat/Augument
augumentpath = os.path.join(path, 'Augment')
# print(augumentpath)
# 遍历文件夹
for root, dirs, files in os.walk(augumentpath):
    # print(root) #当前目录路径
    # print(dirs) #当前路径下所有子目录
    for dir in dirs:
        # print(dir)
        # print(os.path.join(root, dir)) #当前路径下所有子目录
        # 到子目录下遍历找到AugmentData.json
        for root1, dirs1, files1 in os.walk(os.path.join(root, dir)):
            # print(root1) #当前目录路径
            # print(dirs1) #当前路径下所有子目录
            # print(files1) #当前路径下所有非目录子文件
            for file in files1:
                if file == 'AugmentData.json':
                    print(os.path.join(root1, file))
                    target_list.append(os.path.join(root1, file))

            break
    # print(files) #当前路径下所有非目录子文件
    break
complete_id_list = []
all_data=[]
# conversation_dict=[]
i,j,k=0,0,0
m,n=0,0
outofdomain=[]
outofact=[]

rf=open(raw_file, 'r', encoding='utf-8')
raw_data = json.load(rf)
rf.close()

for target in target_list:
    with open(target, 'r', encoding='utf-8') as f:
        datalist = json.load(f)
        for conversation in datalist:
            flag=1
            id=conversation['conversationId']
            for seqid,turn in enumerate(conversation["utterances"]):

                #出现人工修改的话替换回原来的
                if turn=="人工修改":
                    i+=1
                    # flag=0
                    # break
                    #二分查找：找到对应的id
                    for conversation1 in raw_data:
                        if conversation1["conversationId"]==id:
                            turn=conversation1["utterances"][seqid]
                            conversation["utterances"][seqid]=conversation1["utterances"][seqid]
                            break
                #严格清洗
                try:
                    jflag=0
                    kflag=0
                    iflag=0
                    sflag=0
                    for asv in turn["annotation"]:
                        if asv["act_label"] not in ACT:
                            outofact.append(asv["act_label"])
                            n+=1
                            for conversation1 in raw_data:
                                if conversation1["conversationId"]==id:
                                    # turn=conversation1["utterances"][seqid]
                                    conversation["utterances"][seqid]=conversation1["utterances"][seqid]
                                    break
                            break
                            # flag=0
                            # break

                        for dsv in asv["slot_values"]:
                            
                            if dsv["domain"] =="None" or dsv["domain"] =="none" or  dsv["domain"] ==None:
                                # #替换成空字符串
                                # dsv["domain"]=""
                                # 改写价值不大，且增加标签类别，暂时不改
                                for conversation1 in raw_data:
                                    if conversation1["conversationId"]==id:
                                        # turn=conversation1["utterances"][seqid]
                                        conversation["utterances"][seqid]=conversation1["utterances"][seqid]
                                        jflag=1
                                        break
                                if jflag==1:
                                    break
                            if dsv["slot"] =="None" or dsv["slot"] =="none" or dsv["slot"] ==None:
                                # 改写价值不大，且增加标签类别，暂时不改
                                for conversation1 in raw_data:
                                    if conversation1["conversationId"]==id:
                                        # turn=conversation1["utterances"][seqid]
                                        conversation["utterances"][seqid]=conversation1["utterances"][seqid]
                                        kflag=1
                                        break
                                if kflag==1:
                                    break
                            if dsv["value"] =="None" or dsv["value"] =="none" or dsv["value"] ==None:
                                dsv["value"]=""
                            if (dsv["value"]=="" or dsv["value"]=="？") and (dsv["slot"]!="" and dsv["domain"]!=""):
                                dsv["value"]="?"
                            if dsv["domain"] not in DOMAIN:
                                outofdomain.append(dsv["domain"])
                                j+=1
                                for conversation1 in raw_data:
                                    if conversation1["conversationId"]==id:
                                        # turn=conversation1["utterances"][seqid]
                                        conversation["utterances"][seqid]=conversation1["utterances"][seqid]
                                        jflag=1
                                        break
                                if jflag==1:
                                    break
                            if dsv["slot"] not in SLOT:
                                k+=1
                                for conversation1 in raw_data:
                                    if conversation1["conversationId"]==id:
                                        conversation["utterances"][seqid]=conversation1["utterances"][seqid]
                                        kflag=1
                                        break
                                if kflag==1:
                                    break
                            # 判断是否在TAG_VOCAB和INTENT_VOCAB中
                            if dsv["value"] == "?" or dsv["value"] == "":
                                adsv=""
                                adsv="+".join([asv["act_label"],dsv["domain"],dsv["slot"],dsv["value"]])
                                if adsv not in INTENT_VOCAB:
                                    for conversation1 in raw_data:
                                        if conversation1["conversationId"]==id:
                                            conversation["utterances"][seqid]=conversation1["utterances"][seqid]
                                            iflag=1
                                            break
                                if iflag==1:
                                    break
                            else:
                                ads=""
                                ads="+".join(['B',asv["act_label"],dsv["domain"],dsv["slot"]])
                                if ads not in TAG_VOCAB:
                                    for conversation1 in raw_data:
                                        if conversation1["conversationId"]==id:
                                            conversation["utterances"][seqid]=conversation1["utterances"][seqid]
                                            sflag=1
                                            break
                                if sflag==1:
                                    break

                        if jflag==1 or kflag==1 or iflag==1 or sflag==1:
                            break

                        # if dsv["value"]!="" and dsv["value"] not in turn["utterance"]:
                        #     flag=0
                        #     break
                except: 
                    for conversation1 in raw_data:
                        if conversation1["conversationId"]==id:
                            conversation["utterances"][seqid]=conversation1["utterances"][seqid]   
                            break
                    
            if (flag==1):
                # 上述代码没改动flag的原因是，不满足条件的全部改成原始语句。
                all_data.append(conversation)
                complete_id_list.append(id)
                print(len(complete_id_list))
all_data.sort(key=lambda x: x["conversationId"])
complete_id_list.sort()
#统计complete_id_list中每个id出现的次数

c=Counter(complete_id_list)
c=dict(c)
c=sorted(c.items(),key=lambda x:x[1],reverse=True)
print(c)

f=open(os.path.join(path,"complete_id_list_stric.txt"),"w")
for id in complete_id_list:
    f.write(str(id))
    f.write("\n")
f.close()

with open(os.path.join(path,'all_data_stric.json'), 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)
print("需要人工修改计数=",i)
print("domain不在范围内计数=",j)
print("slot不在范围内计数=",k)
print("act不在范围内计数=",n)
 # 计算outofdomain中重复次数
c=Counter(outofdomain)
c=dict(c)
c=sorted(c.items(),key=lambda x:x[1],reverse=True)
print(c)
print()
# 计算outofact中重复次数
c=Counter(outofact)
c=dict(c)
c=sorted(c.items(),key=lambda x:x[1],reverse=True)
print(c)
            
                
                
