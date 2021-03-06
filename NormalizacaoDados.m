% Normalização de dados para executar em uma rede neural Feedforward e Recorrente
% Código feito em Octave

function DadosNormalizados = normalizacaoDados(X)
	Media = mediaDados(X)
	Desvio = desvioPadraoDados(X)
	N = size(X);
	DadosNormalizados=[];
	for i=1: N
		DadosNormalizados(i,:) = (X(i,:)-Media)/Desvio
	end
end

function Media = mediaDados(X)
	sizeX = size(X, 1);
	somaTotal = 0;
	for i=1: sizeX
		somaTotal += X(i,:);
	end
	Media = somaTotal/sizeX;
end

function DesvioPadrao = desvioPadraoDados(X)
	Media = mediaDados(X);
	N = size(X,1);
	DesvioPadrao = 0;
	for i=1: N
		DesvioPadrao += (X(i,:) - Media / N)^2;
	end
	DesvioPadrao = sqrt(DesvioPadrao);
end

function dadosDesnomalizados = desnormalizacaoDados(Media, Desvio, X)
	dadosDesnomalizados = [];
	for i=1: size(X)
		dadosDesnomalizados(i,:) = (X(i,:) * Desvio) + Media;
	end
end

%Dados = [1;2;3;4];
%DadosNormalizados = normalizacaoDados(Dados);
%Media = mediaDados(Dados);
%DesvioPadrao = desvioPadraoDados(Dados);
%DadosDesnormalizados = desnormalizacaoDados(Media, DesvioPadrao, DadosNormalizados)
