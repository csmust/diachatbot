#这个状态结构不再使用，使用state_structure.py
def default_state():
    state = dict(user_action=[],
                 system_action=[],
                 belief_state={},
                 cur_domain=None,
                 request_slots=[],
                 terminated=False,
                 history=[])
    state['belief_state'] = {

    "基本信息": {
        "体重": "",
        "身高": "",
        "年龄": "",
        "既往史": "",
        "性别": ""
    },
    "问题": {
        "血糖值": "",
        "疾病": "",
        "时间": "",
        "影响问题": "",
        "症状": "",
        "持续时长": "",
        "症状部位": "",
        "状态": ""
    },
    "饮食": {
        "饮食名": "",
        "饮食量": "",
        "时间": "",
        "成分": "",
        "成份量": "",
        "影响问题": "",
        "效果": ""
    },
    "运动": {
        "运动名": "",
        "持续时长": "",
        "影响问题": "",
        "强度": "",
        "时间": "",
        "频率": "",
        "效果": ""
    },
    "行为": {
        "行为名": "",
        "持续时长": "",
        "时间": "",
        "影响问题": "",
        "频率": "",
        "效果": ""
    },
    "治疗": {
        "药品": "",
        "用药（治疗）频率": "",
        "用药量": "",
        "持续时长": "",
        "检查项": "",
        "检查值": "",
        "时间": "",
        "部位": "",
        "治疗名": "",
        "影响问题": "",
        "效果": "",
        "药品类型": "",
        "适应症": ""
    }

}
    return state

