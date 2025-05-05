import os
import pytest

from onstove import OnStove

@pytest.mark.order(after="test_model_run.py::test_run_model")
def test_plot_maps():
    # 1. Reading results
    country = 'Rwanda'
    results = OnStove.read_model(os.path.join('onstove', 'tests', 'tests_data', 'output',
                                              'results.pkl'))
                                              
    # 2. Creating result maps
    cmap = {"Biomass ICS (ND)": '#6F4070', "LPG": '#66C5CC', "Biomass": '#FFB6C1',
            "Biomass ICS (FD)": '#af04b3', "Pellets ICS (FD)": '#ef02f5',
            "Charcoal": '#364135', "Charcoal ICS": '#d4bdc5',
            "Biogas": '#73AF48', "Biogas and Biomass ICS (ND)": "#F6029E",
            "Biogas and Biomass ICS (FD)": "#F6029E",
            "Biogas and Pellets ICS (FD)": "#F6029E",
            "Biogas and LPG": "#0F8554", "Biogas and Biomass": "#266AA6",
            "Biogas and Charcoal": "#3B05DF",
            "Biogas and Charcoal ICS": "#3B59DF",
            "Electricity": '#CC503E', "Electricity and Biomass ICS (ND)": "#B497E7",
            "Electricity and Biomass ICS (FD)": "#B497E7",
            "Electricity and Pellets ICS (FD)": "#B497E7",
            "Electricity and LPG": "#E17C05", "Electricity and Biomass": "#FFC107",
            "Electricity and Charcoal ICS": "#660000",
            "Electricity and Biogas": "#f97b72",
            "Electricity and Charcoal": "#FF0000"}

    labels = {"Biogas and Electricity": "Electricity and Biogas",
              'Collected Traditional Biomass': 'Biomass',
              'Collected Improved Biomass': 'Biomass ICS (ND)',
              'Traditional Charcoal': 'Charcoal',
              'Biomass Forced Draft': 'Biomass ICS (FD)',
              'Pellets Forced Draft': 'Pellets ICS (FD)'}
              
    scale = int(results.base_layer.meta['width']//100*10000*2)
    scale_bar_prop = dict(size=scale, style='double', textprops=dict(size=8),
                          linekw=dict(lw=1, color='black'), extent=0.01)
    north_arow_prop = dict(size=30, location=(0.92, 0.92), linewidth=0.5)
              
    results.to_image('max_benefit_tech', name='max_benefit_tech.pdf', cmap=cmap, legend_position=(1, 0.6), 
                     figsize=(7, 5), dpi=300, stats=True, 
                     stats_kwargs={'extra_stats': None, 'fontsize': 10, 'stats_position': (1, 0.9), 'pad': 2, 'sep': 0, 'fontcolor': 'black', 'fontweight': 'normal',
                           'box_props': dict(facecolor='lightyellow', edgecolor='black', alpha=1, boxstyle="sawtooth")},
                     labels=labels, legend=True, 
                     legend_title='Maximum benefit\ncooking technology', 
                     legend_prop={'title': {'size': 10, 'weight': 'bold'}, 
                                  'size': 10},
                     title=f'Maximum benefit technology | {country}',
                     scale_bar=scale_bar_prop, north_arrow=north_arow_prop, 
                     rasterized=True)
    results.to_image('maximum_net_benefit', name='maximum_net_benefit.pdf', cmap='Spectral', 
                     cumulative_count=[0.01, 0.99], figsize=(7, 5),
                     title=f'Maximum net benefit | {country}', dpi=300,
                     rasterized=True)
    assert True
    
@pytest.mark.order(after="test_model_run.py::test_run_model")    
def test_plot_stats():
    # 1. Reading results
    country = 'Rwanda'
    results = OnStove.read_model(os.path.join('onstove', 'tests', 'tests_data', 'output',
                                              'results.pkl'))
                                              
    # 2. Creating result stat plots
    cmap = {"Biomass ICS (ND)": '#6F4070', "LPG": '#66C5CC', "Biomass": '#FFB6C1',
            "Biomass ICS (FD)": '#af04b3', "Pellets ICS (FD)": '#ef02f5',
            "Charcoal": '#364135', "Charcoal ICS": '#d4bdc5',
            "Biogas": '#73AF48', "Biogas and Biomass ICS (ND)": "#F6029E",
            "Biogas and Biomass ICS (FD)": "#F6029E",
            "Biogas and Pellets ICS (FD)": "#F6029E",
            "Biogas and LPG": "#0F8554", "Biogas and Biomass": "#266AA6",
            "Biogas and Charcoal": "#3B05DF",
            "Biogas and Charcoal ICS": "#3B59DF",
            "Electricity": '#CC503E', "Electricity and Biomass ICS (ND)": "#B497E7",
            "Electricity and Biomass ICS (FD)": "#B497E7",
            "Electricity and Pellets ICS (FD)": "#B497E7",
            "Electricity and LPG": "#E17C05", "Electricity and Biomass": "#FFC107",
            "Electricity and Charcoal ICS": "#660000",
            "Electricity and Biogas": "#f97b72",
            "Electricity and Charcoal": "#FF0000"}

    labels = {"Biogas and Electricity": "Electricity and Biogas",
              'Collected Traditional Biomass': 'Biomass',
              'Collected Improved Biomass': 'Biomass ICS (ND)',
              'Traditional Charcoal': 'Charcoal',
              'Biomass Forced Draft': 'Biomass ICS (FD)',
              'Pellets Forced Draft': 'Pellets ICS (FD)'}
              
    results.plot_split(cmap=cmap, labels=labels, save_as='tech_split.pdf', 
                       height=1.5, width=3.5)
    results.plot_costs_benefits(labels=labels, save_as='benefits_costs.png', 
                                height=1.5, width=5, dpi=300)
#     results.plot_distribution(type='histogram', groupby='None',
#                               hh_divider=1000,
#                               y_title='Households (thousands)', cmap=cmap,
#                               labels=labels, save_as='max_benefits_hist.svg',
#                               height=1.5, width=3.5, quantiles=True)
    assert True