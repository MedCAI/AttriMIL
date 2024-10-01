# AttriMIL: Revisiting attention-based multiple instance learning for whole-slide pathological image classification from a perspective of instance attributes
The official implementation of AttriMIL (Pending).

## 1. Introduction
### 1.1 Background
WSI classification typically requires the MIL framework to perform two key tasks: bag classification and instance discrimination, which correspond to clinical diagnosis and the localization of disease-positive regions, respectively. Among various MIL architectures, attention-based MIL frameworks address both tasks simultaneously under weak supervision and thus dominate pathology image analysis. However, attention-based MIL frameworks face two challenges:

(i) The incorrect measure of pathological attributes based on attention, which may confuse diagnosis.
(ii) The negligence of modeling intra-slide and inter-slide interaction, which is essential to obtain robust semantic representation of instances.
<p align="center">
    <img src="./visualization/AttributeScoring.png"/ width="800"> <br />
    <em> 
    Figure 1. Illustration of various viewpoints in colonoscopy images caused by different orientations of the colonoscope tip.
    </em>
</p>


To overcome these issues, we propose a novel framework named attribute-aware multiple instance learning (AttriMIL) tailored for pathological image classification. 

