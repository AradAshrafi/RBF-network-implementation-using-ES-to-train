functionality of this program :

 - [x] Single-Layer Radial Basis Function (RBF) Architecture Implementation 
 
    X contains inputs <br/>
    X =[ (X<sup>1</sup>) &nbsp; ... &nbsp;(X<sup>L</sup>)]<br/>
    G is matrix which contains g<sub>i</sub>(X<sup>j</sup>) (which are outputs of each RBF node) <br/>
    **g<sub>i</sub> = e <sup>(-‫γ‬<sub>i</sub> (X - V<sub>i</sub>)^T(X-V<sub>i</sub>))**</sup> &nbsp; &nbsp; &nbsp; ‫γ<sub>i</sub>‬ is i<sup>th</sup> constant and V<sub>i</sub> is i<sup>th</sup> center <br/>
    G = [ g<sub>1</sub>(X<sup>1</sup>) &nbsp; ... &nbsp; g<sub>m</sub>(X<sup>1</sup>)<br/>
        &nbsp; &nbsp; &nbsp; &nbsp; .<br/> 
        &nbsp; &nbsp; &nbsp; &nbsp; .<br/>
        &nbsp; &nbsp; &nbsp; &nbsp; .     
        &nbsp; &nbsp; &nbsp; &nbsp; g<sub>1</sub>(X<sup>L</sup>) ... g<sub>m</sub>(X<sup>L</sup>)  ]<br/>
    W is a vector contains weights between RBF layer and output <br/>
    W =[ (w<sub>1</sub>) <br/>
        &nbsp; &nbsp; &nbsp; &nbsp; .<br/> 
        &nbsp; &nbsp; &nbsp; &nbsp; .<br/>
        &nbsp; &nbsp; &nbsp; &nbsp; .  
        &nbsp; &nbsp; &nbsp; &nbsp; (w<sub>m</sub>) ]<br/>
    **‫̂y‬= GW <br/>**
 - [x] Single-Layer RBF weights calculation
        
      **W = (G<sup>T</sup> G)<sup>-1</sup> G<sup>T</sup> y**
 - [x] Single-Layer RBF error calculation
 
    L is our loss function (to calculate error) <br/>
    ‫‪L(ŷ,y) = 1/2 - (ŷ - y)<sup>T</sup>(ŷ - y)
 - [x] Evolution Strategy algorithm with V(vector) and ‫‫‫‫‫‫‫γ(scalar) parameters
 - [ ] Show Results with current Architecture for Regression Problem
 - [ ] Show Results with current Architecture for Classification Problem
