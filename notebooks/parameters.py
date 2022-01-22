# fake_trace_datasets = {
#     'nanopw': {
#         'f2pw': ('/mnt/SCA1/TCHES2/data/devices/fake_nanopw-f2.h5','FAKE/traces_08bnanopw'),
#         'f4pw': ('/mnt/SCA1/TCHES2/data/devices/fake_nanopw-f4.h5','FAKE/traces_08bnanopw')
#     },
#     'f2em': { 'f2pw': ('/mnt/SCA1/christophe_test/sca_multi_channels/fake_dataset2/segan/f2em-to-f2pw/fake_dataset_rmsprop_tanh_128_200_snr_0.001vs1.h5','FAKE/traces_08bf2em') },
# #     'f3em': { 'f3pw': ('/mnt/SCA1/TCHES2/data/channels/fake_f3em.h5','FAKE/traces_08bf3em') },
#     'f3em': { 'f3pw': ('/mnt/SCA1/christophe_test/sca_multi_channels/fake_dataset2/segan/f3em-to-f3pw/fake_dataset_rmsprop_tanh_128_200_snr_0.001vs1.h5','FAKE/traces_08bf3em') },
#     'f4em': { 'f4pw': ('/mnt/SCA1/christophe_test/sca_multi_channels/fake_dataset2/segan/f4em-to-f4pw/fake_dataset_rmsprop_tanh_128_200_snr_0.001vs1.h5','FAKE/traces_08bf4em') },
# }


fake_trace_datasets = {
    'f0em': { 
        'f0pw': ('/mnt/SCA1/CARDIS/script/fake_data/fake_08BF0EM+08BF0PW.h5','FAKE/traces_08BF0EM')
    },
    'f0pw': { 
        'f1pw': ('/mnt/SCA1/CARDIS/script/fake_data/fake_08BF0PW+08BF1PW.h5','FAKE/traces_08BF0PW'),
        'f2pw': ('/mnt/SCA1/CARDIS/script/fake_data/fake_08BF0PW+08BF2PW.h5','FAKE/traces_08BF0PW'),
        'f4pw': ('/mnt/SCA1/CARDIS/script/fake_data/fake_08BF0PW+08BF4PW.h5','FAKE/traces_08BF0PW')
    },
    'f1em': { 'f1pw': ('/mnt/SCA1/CARDIS/script/fake_data/fake_08BF1EM+08BF1PW.h5','FAKE/traces_08BF1EM')},
    'f1pw': { 
             'f0pw': ('/mnt/SCA1/CARDIS/script/fake_data/fake_08BF1PW+08BF0PW.h5','FAKE/traces_08BF1PW'),
            'f2pw': ('/mnt/SCA1/CARDIS/script/fake_data/fake_08BF1PW+08BF2PW.h5','FAKE/traces_08BF1PW') ,
             'f4pw': ('/mnt/SCA1/CARDIS/script/fake_data/fake_08BF1PW+08BF4PW.h5','FAKE/traces_08BF1PW') 
      },
    'f2em': { 
        'f2pw': ('/mnt/SCA1/CARDIS/script/fake_data/fake_08BF2EM+08BF2PW.h5','FAKE/traces_08BF2EM') },
    'f2pw': { 
         'f0pw': ('/mnt/SCA1/CARDIS/script/fake_data/fake_08BF2PW+08BF0PW.h5','FAKE/traces_08BF2PW') ,
         'f1pw': ('/mnt/SCA1/CARDIS/script/fake_data/fake_08BF2PW+08BF1PW.h5','FAKE/traces_08BF2PW') ,
         'f3pw': ('/mnt/SCA1/CARDIS/script/fake_data/fake_08BF2PW+08BF3PW.h5','FAKE/traces_08BF2PW') ,
         'f4pw': ('/mnt/SCA1/CARDIS/script/fake_data/fake_08BF2PW+08BF4PW.h5','FAKE/traces_08BF2PW') 
    },
    'f3em': { 
        'f3pw': ('/mnt/SCA1/CARDIS/script/fake_data/fake_08BF3EM+08BF3PW.h5','FAKE/traces_08BF3EM') 
    },
    
    'f4em': { 
        'f4pw': ('/mnt/SCA1/CARDIS/script/fake_data/fake_08BF4EM+08BF4PW.h5','FAKE/traces_08BF4EM') 
    },
     'f4pw': { 
        'f0pw': ('/mnt/SCA1/CARDIS/script/fake_data/fake_08BF4PW+08BF0PW.h5','FAKE/traces_08BF4PW') ,
        'f1pw': ('/mnt/SCA1/CARDIS/script/fake_data/fake_08BF4PW+08BF1PW.h5','FAKE/traces_08BF4PW') ,
        'f2pw': ('/mnt/SCA1/CARDIS/script/fake_data/fake_08BF4PW+08BF2PW.h5','FAKE/traces_08BF4PW') 
     },
}


# learn_trace_datasets = {
#     'data': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','LEARN/data'),
#     'f2em': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','LEARN/traces_08bf2em'),
#     'f2pw': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','LEARN/traces_08bf2pw'),
#     'f3em': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','LEARN/traces_08bf3em'),
#     'f3pw': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','LEARN/traces_08bf3pw'),
#     'f4em': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','LEARN/traces_08bf4em'),
#     'f4pw': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','LEARN/traces_08bf4pw'),
#     'nanopw': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','LEARN/traces_08bnanopw'),
# }

attack_trace_datasets = {
    'data': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','ATTACK/data'),
    'f0em': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','ATTACK/traces_08bf0em'),
    'f0pw': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','ATTACK/traces_08bf0pw'),
    'f1em': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','ATTACK/traces_08bf1em'),
    'f1pw': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','ATTACK/traces_08bf1pw'),
    'f2em': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','ATTACK/traces_08bf2em'),
    'f2pw': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','ATTACK/traces_08bf2pw'),
    'f3em': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','ATTACK/traces_08bf3em'),
    'f3pw': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','ATTACK/traces_08bf3pw'),
    'f4em': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','ATTACK/traces_08bf4em'),
    'f4pw': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','ATTACK/traces_08bf4pw'),
    'nanopw': ('/mnt/SCA1/christophe_test/sca_multi_channels/trace_BD_with_label_SNR4_700_PoI_frame1.h5','ATTACK/traces_08bnanopw'),
}