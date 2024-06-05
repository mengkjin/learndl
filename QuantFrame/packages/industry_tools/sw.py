import pandas as pd
import os


def from_old_to_new(info_):
    assert isinstance(info_, (pd.DataFrame,))
    info = info_.copy()  # type: pd.DataFrame
    info["industry"] = info["swInd1"]

    flag = info["swInd1"] == "申万黑色金属"
    info.loc[flag, "industry"] = "申万钢铁"
    flag = info["swInd1"] == "申万电子元器件"
    info.loc[flag, "industry"] = "申万电子"
    flag = info["swInd1"] == "申万餐饮旅游"
    info.loc[flag, "industry"] = "申万休闲服务"

    flag = info["swInd2"] == "申万建筑材料"
    info.loc[flag, "industry"] = "申万建筑材料"
    flag = info["swInd2"] == "申万建筑装饰"
    info.loc[flag, "industry"] = "申万建筑装饰"
    flag = (info["swInd1"] == "申万机械设备") & (info["swInd2"] == "申万电气设备")
    info.loc[flag, "industry"] = "申万电气设备"
    flag = (info["swInd1"] == "申万交运设备") & (info["swInd2"] == "申万非汽车交运设备") & (info["swInd3"] == "申万铁路设备")
    info.loc[flag, "industry"] = "申万机械设备"
    flag = (info["swInd1"] == "申万交运设备") & (info["swInd2"] == "申万非汽车交运设备") & (
            (info["swInd3"] == "申万航空航天设备") | (info["swInd3"] == "申万船舶制造"))
    info.loc[flag, "industry"] = "申万国防军工"
    flag = (info["swInd1"] == "申万交运设备") & (
            (info["swInd2"] == "申万汽车整车") | (info["swInd2"] == "申万汽车零部件") | (info["swInd2"] == "申万交运设备服务"))
    info.loc[flag, "industry"] = "申万汽车"
    flag = (info["swInd1"] == "申万交运设备") & (info["swInd2"] == "申万非汽车交运设备") & (
            (info["swInd3"] == "申万其他交运设备") | (info["swInd3"] == "申万摩托车") | (info["swInd3"] == "申万SW850932") | (
            info["swInd3"] == "申万农机设备"))
    info.loc[flag, "industry"] = "申万汽车"
    flag = (info["swInd1"] == "申万信息服务") | (info["swInd2"] == "申万通信运营")
    info.loc[flag, "industry"] = "申万通信"
    flag = (info["swInd1"] == "申万信息服务") & ((info["swInd2"] == "申万传媒") | (info["swInd2"] == "申万网络服务"))
    info.loc[flag, "industry"] = "申万传媒"
    flag = (info["swInd1"] == "申万信息服务") & (info["swInd2"] == "申万计算机应用")
    info.loc[flag, "industry"] = "申万计算机"
    flag = (info["swInd1"] == "申万信息设备") & (info["swInd2"] == "申万通信设备")
    info.loc[flag, "industry"] = "申万通信"
    flag = (info["swInd1"] == "申万信息设备") & ((info["swInd2"] == "申万计算机设备") | (info["swInd2"] == "申万计算机应用"))
    info.loc[flag, "industry"] = "申万计算机"
    flag = (info["swInd1"] == "申万信息服务") & (info["swInd2"] == "申万显示器件")
    info.loc[flag, "industry"] = "申万电子"
    flag = info["swInd2"] == "申万视听器材"
    info.loc[flag, "industry"] = "申万家用电器"
    flag = info["swInd2"] == "申万塑料"
    info.loc[flag, "industry"] = "申万化工"
    rtn = list(info["industry"])
    return rtn


def from_new_to_old(info_):
    assert isinstance(info_, (pd.DataFrame,))
    info = info_.copy()  # type: pd.DataFrame
    info["industry"] = info["swInd1"]

    flag = info["swInd1"] == "申万黑色金属"
    info.loc[flag, "industry"] = "申万钢铁"
    flag = info["swInd1"] == "申万电子元器件"
    info.loc[flag, "industry"] = "申万电子"
    flag = info["swInd1"] == "申万餐饮旅游"
    info.loc[flag, "industry"] = "申万休闲服务"

    flag = (info["swInd1"] == "申万建筑材料") | (info["swInd1"] == "申万建筑装饰")
    info.loc[flag, "industry"] = "申万建筑建材"
    flag = info["swInd1"] == "申万电气设备"
    info.loc[flag, "industry"] = "申万机械设备"
    flag = (info["swInd1"] == "申万机械设备") & (info["swInd2"] != "申万运输设备")
    info.loc[flag, "industry"] = "申万机械设备"
    flag = (info["swInd1"] == "申万国防军工") | (info["swInd1"] == "申万汽车") | (
        (info["swInd1"] == "申万机械设备") & (info["swInd2"] == "申万运输设备"))
    info.loc[flag, "industry"] = "申万交运设备"
    flag = (info["swInd1"] == "申万计算机") & (info["swInd2"] == "申万计算机应用")
    info.loc[flag, "industry"] = "申万信息服务"
    flag = (info["swInd1"] == "申万计算机") & (info["swInd2"] == "申万计算机设备")
    info.loc[flag, "industry"] = "申万信息设备"
    flag = info["swInd1"] == "申万传媒"
    info.loc[flag, "industry"] = "申万信息服务"
    flag = (info["swInd1"] == "申万通信") & (info["swInd2"] == "申万通信运营")
    info.loc[flag, "industry"] = "申万信息服务"
    flag = (info["swInd1"] == "申万通信") & (info["swInd2"] == "申万通信设备")
    info.loc[flag, "industry"] = "申万信息设备"
    rtn = list(info["industry"])
    return rtn


def from_ch_to_en(ch_nm_list_, ind_version_="new"):
    assert ind_version_ in ["backup", "new"]
    old_name_dict = {"申万商业贸易": "CommercialTrade", "申万食品饮料": "FoodAndBeverage",
                     "申万信息服务": "InformationService", "申万交运设备": "TransportationEquipment",
                     "申万农林牧渔": "Agriculture", "申万轻工制造": "LightManufacturing",
                     "申万银行": "Bank", "申万有色金属": "DiversifiedMetals",
                     "申万综合": "Synthetics", "申万信息设备": "InformationEquipment",
                     "申万休闲服务": "LeisureServices", "申万医药生物": "MedicationAndBioIndustry",
                     "申万纺织服装": "TextileAndGarment", "申万非银金融": "NonBankFinancial",
                     "申万钢铁": "Metal", "申万采掘": "ExtractiveIndustries",
                     "申万电子": "ElectronicIndustry", "申万房地产": "RealEstate",
                     "申万家用电器": "HouseholdAppliances", "申万建筑建材": "ConstructionAndMaterials",
                     "申万交通运输": "CommunicationsAndTransportation", "申万公用事业": "PublicUtility",
                     "申万化工": "ChemicalIndustry", "申万机械设备": "Machinery"}
    new_name_dict = {"申万商业贸易": "CommercialTrade", "申万食品饮料": "FoodAndBeverage",
                     "申万计算机": "Computer", "申万传媒": "MultiMedia",
                     "申万国防军工": "DefenseIndustry", "申万电气设备": "ElectricalEquip",
                     "申万汽车": "MotorVehicle", "申万通信": "Telecoms",
                     "申万农林牧渔": "Agriculture", "申万轻工制造": "LightManufacturing",
                     "申万银行": "Bank", "申万有色金属": "DiversifiedMetal",
                     "申万综合": "Synthetics", "申万休闲服务": "LeisureServices",
                     "申万医药生物": "MedicationAndBio", "申万纺织服装": "TextileAndGarment",
                     "申万非银金融": "NonBankFinancial", "申万钢铁": "Metal",
                     "申万采掘": "ExtractiveIndustry", "申万电子": "ElectronicIndustry",
                     "申万房地产": "RealEstate", "申万家用电器": "HouseholdAppliances",
                     "申万建筑材料": "ConstructionAndMaterial", "申万建筑装饰": "BuildingDecoration",
                     "申万交通运输": "CommunicationsAndTransportation", "申万公用事业": "PublicUtility",
                     "申万化工": "ChemicalIndustry", "申万机械设备": "Machinary"}
    eng_inds = list()

    if ind_version_ == "backup":
        name_dict = old_name_dict
    elif ind_version_ == "new":
        name_dict = new_name_dict
    else:
        assert False
    for ch_nm in ch_nm_list_:
        eng_inds.append(name_dict.get(ch_nm, "null"))
    return eng_inds


def from_en_to_id(en_nm_list_):
    # only implemented for the new industry classification version
    industry_info = pd.DataFrame([("Agriculture", 1),
                                  ("BuildingDecoration", 2),
                                  ("ChemicalIndustry", 3),
                                  ("CommercialTrade", 4),
                                  ("CommunicationsAndTransportation", 5),
                                  ("Computer", 6),
                                  ("ConstructionAndMaterial", 7),
                                  ("DefenseIndustry", 8),
                                  ("DiversifiedMetal", 9),
                                  ("ElectricalEquip", 10),
                                  ("ElectronicIndustry", 11),
                                  ("ExtractiveIndustry", 12),
                                  ("FoodAndBeverage", 13),
                                  ("HouseholdAppliances", 14),
                                  ("LeisureServices", 15),
                                  ("LightManufacturing", 16),
                                  ("Machinary", 17),
                                  ("MedicationAndBio", 18),
                                  ("Metal", 19),
                                  ("MotorVehicle", 20),
                                  ("MultiMedia", 21),
                                  ("NonBankFinancial", 22),
                                  ("PublicUtility", 23),
                                  ("RealEstate", 24),
                                  ("Synthetics", 25),
                                  ("Telecoms", 26),
                                  ("TextileAndGarment", 27),
                                  ("Bank", 28)], columns=["newindustry", "industryID"])

    df = pd.merge(
        pd.DataFrame(en_nm_list_, columns=["newindustry"]),
        industry_info,
        how="left", on=["newindustry"]
    )
    rtn = df["industryID"].fillna(-1).tolist()
    return rtn


def from_ch_to_id(ch_nm_list_):
    en_nm = from_ch_to_en(ch_nm_list_)
    rtn = from_en_to_id(en_nm)
    return rtn


