import onsstove.technology


specs_data = pd.read_excel('path_to_file', sheet_name='CountryData')

country = specs_data.iloc[0]['Country']
iso_code = specs_data.iloc[0]['iso_code']
start_year = float(specs_data.iloc[0]['StartYear'])
end_year = float(specs_data.iloc[0]['EndYear'])
pop_start = float(specs_data.iloc[0]['PopStart'])
urb_start = float(specs_data.iloc[0]['UrbStart'])
elec_start_national = float(specs_data.iloc[0]['ElecStartNat'])
elec_start_urban = float(specs_data.iloc[0]['ElecStartUrb'])
elec_start_rural = float(specs_data.iloc[0]['ElecStartRur'])
clean_cook_start = float(specs_data.iloc[0]['CleanCookStart'])
clean_cook_urban = float(specs_data.iloc[0]['CleanCookUrb'])
clean_cook_rural = float(specs_data.iloc[0]['CleanCookRur'])
pop_end = float(specs_data.iloc[0]['PopEnd'])
urb_end = float(specs_data.iloc[0]['UrbEnd'])
min_wage = float(specs_data.iloc[0]['MinWage'])
hhsize_rural = float(specs_data.iloc[0]['NumPeoplePerHHRural'])
hhsize_urban = float(specs_data.iloc[0]['NumPeoplePerHHUrban'])
copd_mort = float(specs_data.iloc[0]['COPDMort'])
lc_mort = float(specs_data.iloc[0]['LCMort'])
ihd_mort = float(specs_data.iloc[0]['IHDMort'])
alri_mort = float(specs_data.iloc[0]['ALRIMort'])
copd_prev = float(specs_data.iloc[0]['COPDPrev'])
lc_prev = float(specs_data.iloc[0]['LCPrev'])
ihd_prev = float(specs_data.iloc[0]['IHDPrev'])
alri_prev = float(specs_data.iloc[0]['ALRIPrev'])
copd_coi = float(specs_data.iloc[0]['COPDCOI'])
lc_coi = float(specs_data.iloc[0]['LCCOI'])
ihd_coi = float(specs_data.iloc[0]['IHDCOI'])
alri_coi = float(specs_data.iloc[0]['AKRICOI'])

class OnSSTOVE():
    '''
    This is the main class of OnSTOVE. It contains all methods and parameters needed
    to run a Clean Cooking Access model in any geography
    '''
    population = 'path_to_file'
    def __init__(self):
        '''
        Initialization of the class
        '''
        pass