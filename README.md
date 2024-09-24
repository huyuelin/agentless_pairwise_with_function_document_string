基于agentless在函数定位这一步做了优化，结合了pairwise的思想（根据不同文件名分批次喂给llm，每次Prompt的token量减少）以及在函数定位时加上每个函数的document_string，同时在函数定位准确性上基于原始版本的agentless提升了10%。
![image](https://github.com/user-attachments/assets/d7ebc069-c7d7-4450-833b-c6fe445af734)
