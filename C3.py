from pandas import DataFrame as df
import numpy as np

class Photosynthesis_C3_ACi():
    def __init__(self,input,photosyntheis):
        self.input = input
        self.photosynthesis = photosyntheis
        constant = {'R' : 8.314/1000} 
        self.constants = constant
    
    def initialze_ACi(self):
        LeafState = df()
        LeafState['ci'] = self.input.loc[:,'Ci'] # Intercellular concentration of CO2 in air corrected for solubility relative to 25 degree Celcius [ppm]
        LeafState['cm'] = self.input.loc[:,'Ci']*0.8
        LeafState['gm'] = self.input.loc[:,'Ci']*0
        LeafState['gammaStar'] = self.input.loc[:,'Ci']*0 + 0.000193 #  Half of inverse of rubisco specifiity [-] 
        LeafState['temperature'] = self.input.loc[:,'temperature']

        LeafMassFlux = df()
        LeafMassFlux['tpu'] = self.input.loc[:,'Ci']*0
        LeafMassFlux['ac'] = self.input.loc[:,'Ci']*0 #1 # RUBISCO limited gross rate of CO2 uptake per unit area [u moles m-2 s-1]
        LeafMassFlux['aj'] = self.input.loc[:,'Ci']*0 #1 # TPU limited gross rate of CO2 uptake per unit area [u moles m-2 s-1]
        LeafMassFlux['ap'] = self.input.loc[:,'Ci']*0 #1 # RuBP -limited gross rate of CO2 uptake per unit area [u moles m-2 s-1]
        LeafMassFlux['aGross'] = self.input.loc[:,'Ci']*0 #Weather.ca/10#1 # Gross rate of CO2 uptake per unit area [u moles m-2 s-1]
        LeafMassFlux['aNet'] = self.input.loc[:,'Ci']/10  # Net rate of CO2 uptake [u mol m-2 s-1]
        LeafMassFlux['J'] = self.input.loc[:,'Ci']*0 #1 # Whole chain electron transport rate (vonCaemmerer 2000) [u moles m-2 s-1]
        LeafMassFlux['rd'] = self.input.loc[:,'Ci']*0 + self.photosynthesis['rd25'] #1 # Dark respiration at leaf (vonCaemmerer 2000) [u moles m-2 s-1]
        LeafMassFlux['vcmax'] = self.input.loc[:,'Ci']*0 #Photosynthesis.vcmax25#1 # Max RuBP saturated carboxylation at given temperature (Massad 2007) [u moles m-2 s-1]
        LeafMassFlux['jmax'] = self.input.loc[:,'Ci']*0 #Photosynthesis.jmax25#1 # Max electon transport rate (Massad 2007) [u moles m-2 s-1]
        self.leafstate = LeafState
        self.leafmassflux = LeafMassFlux
        return LeafMassFlux,LeafState
    
    def C3_Aci(self):
        LeafMassFlux = self.leafmassflux.loc[:,:]
        LeafState = self.leafstate.loc[:,:]
        Photosynthesis = self.photosynthesis
        Constants = self.constants
        Weather = self.input.loc[:,:]

        LeafTemperatureKelvin = LeafState.loc[:,'temperature'] + 273.15 # [K]
        LeafState.loc[:,'gm'] = Photosynthesis['gm25']*np.exp(20.01 - (49600 / (Constants['R'] * 1000.0 * LeafTemperatureKelvin)))/(1+np.exp((1400 * LeafTemperatureKelvin - 437400) / (Constants['R'] * 1000.0 * LeafTemperatureKelvin))) # [m mol m-2 s-1]

        LeafState.loc[:,'cm'] = LeafState.loc[:,'ci'] - LeafMassFlux.loc[:,'aNet']/LeafState.loc[:,'gm'] # [micro mol mo-1]
    
        Ko = 1000/Weather.loc[:,'pressure']*np.exp(12.3772 - 23720/(Constants['R'] * 1000.0 * LeafTemperatureKelvin))

        Kc = 1000/Weather.loc[:,'pressure']*np.exp(35.9774 - 80990 /(Constants['R'] * 1000.0 * LeafTemperatureKelvin)) # [mu mol mol-1]

        # Compute respiration
        LeafMassFlux.loc[:,'rd'] = Photosynthesis['rd25']*np.exp(18.7145 - 46.39 /(Constants['R'] * LeafTemperatureKelvin))  # (Bernacchi 2003)) [mu mol m-2 s-1]

        # Temperature response for Vcmax
        LeafMassFlux.loc[:,'vcmax'] = Photosynthesis['vcmax25'] * np.exp(26.355 - 65.33 / (Constants['R'] * LeafTemperatureKelvin)) # (Bernacchi 2003) [mu mol m-2 s-1]

        # Temperature response for CO2 compensation point
        LeafState.loc[:,'gammaStar'] = 1000/Weather.loc[:,'pressure']*np.exp(11.187 - 24460/(Constants['R'] * 1000.0 * LeafTemperatureKelvin)) # [mu mol mol-1]

        LeafMassFlux.loc[:,'J'] = Photosynthesis['jmax25'] * np.exp(17.71 - 43.9 / (Constants['R'] * LeafTemperatureKelvin)) # (Bernacchi 2003)

        LeafMassFlux.loc[:,'aj'] = (1.0 - LeafState.loc[:,'gammaStar'] / LeafState.loc[:,'cm']) * (LeafMassFlux.loc[:,'J'] * LeafState.loc[:,'cm']) /(4 * LeafState.loc[:,'cm'] + 8 * LeafState.loc[:,'gammaStar']) # Light limited photosynthesis (vonCaemmerer 2000)

        # CO2 limited photosynthesis
        LeafMassFlux.loc[:,'ac'] = (1.0 - LeafState.loc[:,'gammaStar'] / LeafState.loc[:,'cm']) * (LeafMassFlux.loc[:,'vcmax'] * LeafState.loc[:,'cm']) /(LeafState.loc[:,'cm'] + Kc * (1.0 + Weather.loc[:,'O2'] / Ko)) # Rubisco limited photosynthesis (vonCaemmerer 2000)

        # TPU limited photosynthesis
        LeafMassFlux.loc[:,'tpu'] = Photosynthesis['tpu25']*np.exp(21.46 - (53100 / (Constants['R'] * 1000.0 * LeafTemperatureKelvin))) \
            /(1+np.exp((650 * LeafTemperatureKelvin - 201800) / (Constants['R'] * 1000.0 * LeafTemperatureKelvin))) # [m mol m-2 s-1]

        LeafMassFlux.loc[:,'ap'] = (3.0 * LeafMassFlux.loc[:,'tpu']) # TPU limited photosynthesis (vonCaemmerer 2000)

        LeafMassFlux.loc[:,'aGross'] = np.min([LeafMassFlux.loc[:,'aj'],LeafMassFlux.loc[:,'ac'],LeafMassFlux.loc[:,'ap']])
        LeafMassFlux.loc[:,'aNet'] = LeafMassFlux.loc[:,'aGross'] - LeafMassFlux.loc[:,'rd']

        self.leafmassflux = LeafMassFlux
        self.leafstate = LeafState
        
        return LeafMassFlux,LeafState
    
    def solve(self):
        self.initialze_ACi()
        self.C3_Aci()
        return self.leafmassflux,self.leafstate

                
        