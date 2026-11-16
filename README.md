Detecção de Intrusões em Tráfego IoT com Autoencoders — AE-PECL

Repositório dos códigos desenvolvidos para o Trabalho de Conclusão de Curso em Engenharia de Computação, intitulado “Detecção de Intrusões na Borda de Redes com Aprendizagem de Máquina Não Supervisionada”, da Universidade do Vale do Itajaí (UNIVALI).

Sobre o projeto

Este trabalho apresenta o desenvolvimento e a avaliação de uma estratégia adaptada de treinamento de Autoencoders para detecção de intrusões em tráfego de dispositivos IoT.

A abordagem proposta, denominada AE-PECL (Autoencoder com Perturbação Estruturada e Consistência Latente), combina reconstrução convencional, reconstrução de entradas perturbadas e regularização das representações latentes, buscando melhorar a identificação de comportamentos maliciosos.

Metodologia

Os experimentos utilizam o conjunto de dados CICIoT2023, com treinamento baseado exclusivamente em registros benignos. A avaliação compara o Autoencoder Convencional de Referência (AEC-REF) com o modelo adaptado AE-PECL-010, utilizando a mesma arquitetura de 3.268 parâmetros treináveis.

Foram investigadas diferentes configurações de treinamento, limiares de detecção e sementes de inicialização.

Principais resultados

Na avaliação final, realizada com 319.242 registros, o AE-PECL-010 apresentou Recall de 54,80% e F1-score de 66,73%, superando os valores de 53,16% e 65,29% obtidos pelo AEC-REF.

A adaptação proporcionou uma redução relativa de aproximadamente 3,49% dos falsos negativos, correspondente a um ganho líquido de 2.705 ataques detectados.

Esses resultados representam uma melhoria experimental moderada nas condições avaliadas, sem comprovar a viabilidade operacional do modelo em equipamentos de borda.

Autoria

Guilherme da Silva Santana
