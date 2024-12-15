    # while True:
    #     data = server.get_rigidbody_set_frame()
    #     data2 = server.get_marker_set_frame()
    #     if data is not None and data2 is not None:
    #         markerset_name_list = data2.marker_set_dict.keys()
    #         for i,v in enumerate(markerset_name_list):
    #             if v == "others":
    #                 continue
    #             rigidbodydata = data.rigidbody_set_dict[i+1]
    #             print(f"{v} : {rigidbodydata}")
