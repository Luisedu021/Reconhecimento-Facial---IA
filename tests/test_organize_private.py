"""Organização das fotos privadas: só com imagens SINTÉTICAS (quadrados coloridos) em pastas temporárias."""

import csv
import os

import pytest
from PIL import Image

from facerec.config import ROOT
from facerec.organize_private import organize


def make_photos(folder, names_and_times):
    folder.mkdir(parents=True)
    for name, t in names_and_times:
        path = folder / name
        Image.new("RGB", (8, 8), (t % 255, 0, 0)).save(path)
        os.utime(path, (t, t))  # sem EXIF: a ordem vem da data do arquivo


def test_move_para_ids_pseudonimos_e_divide_por_data(tmp_path):
    origem = tmp_path / "dataset"
    # Nomes de arquivo fora de ordem de propósito: quem manda é a data.
    make_photos(origem / "Fulana", [("z.jpg", 1000), ("a.jpg", 2000), ("m.jpg", 3000), ("b.jpg", 4000), ("c.jpg", 5000)])
    make_photos(origem / "Beltrano", [(f"{i}.jpg", 100 * i) for i in range(1, 6)])
    destino, mapa = tmp_path / "private", tmp_path / "fora" / "mapa.csv"

    summary = organize(origem, destino, mapa)

    assert summary == {"P01": {"S1": 2, "S2": 3, "nao_suportadas": 0}, "P02": {"S1": 2, "S2": 3, "nao_suportadas": 0}}
    # Ordem alfabética das pastas: Beltrano -> P01, Fulana -> P02.
    with open(mapa, encoding="utf-8") as f:
        assert list(csv.reader(f))[1:] == [["Beltrano", "P01"], ["Fulana", "P02"]]
    # As duas fotos mais antigas da Fulana (z.jpg, a.jpg) viraram o cadastro S1.
    s1 = sorted(p.name for p in (destino / "P02" / "S1").iterdir())
    assert s1 == ["1.jpg", "2.jpg"]
    # Cor de cada foto = data % 255 (235, 215, 195, ...); o JPEG pode mudar 1-2 níveis, por isso a tolerância.
    assert abs(Image.open(destino / "P02" / "S1" / "1.jpg").getpixel((0, 0))[0] - 1000 % 255) <= 3
    assert abs(Image.open(destino / "P02" / "S2" / "1.jpg").getpixel((0, 0))[0] - 3000 % 255) <= 3
    # Nenhum nome de pessoa nos caminhos de destino, e a origem foi esvaziada e removida.
    assert not any("Fulana" in str(p) or "Beltrano" in str(p) for p in destino.rglob("*"))
    assert not origem.exists()


def test_mapa_dentro_do_repositorio_e_recusado(tmp_path):
    make_photos(tmp_path / "dataset" / "Fulana", [("a.jpg", 1)])
    with pytest.raises(ValueError):
        organize(tmp_path / "dataset", tmp_path / "private", ROOT / "mapa.csv")
    with pytest.raises(ValueError):
        organize(tmp_path / "dataset", tmp_path / "private", tmp_path / "OneDrive - X" / "mapa.csv")
    assert (tmp_path / "dataset" / "Fulana" / "a.jpg").exists()  # nada foi movido


def test_fotos_soltas_param_sem_mover_nada(tmp_path):
    origem = tmp_path / "dataset"
    make_photos(origem, [("a.jpg", 1), ("b.jpg", 2)])
    with pytest.raises(SystemExit, match="2 foto"):
        organize(origem, tmp_path / "private", tmp_path / "fora" / "mapa.csv")
    assert (origem / "a.jpg").exists() and not (tmp_path / "private").exists()


def test_nao_sobrescreve_participantes_existentes(tmp_path):
    make_photos(tmp_path / "dataset" / "Fulana", [("a.jpg", 1)])
    make_photos(tmp_path / "private" / "P01" / "S1", [("1.jpg", 1)])
    with pytest.raises(ValueError):
        organize(tmp_path / "dataset", tmp_path / "private", tmp_path / "fora" / "mapa.csv")


def test_fotos_em_subpastas_e_arquivos_nao_suportados(tmp_path):
    make_photos(tmp_path / "dataset" / "Fulana" / "dia1", [("a.jpg", 1), ("b.jpg", 2)])
    make_photos(tmp_path / "dataset" / "Fulana" / "dia2", [("c.png", 3)])
    (tmp_path / "dataset" / "Fulana" / "dia2" / "d.heic").write_bytes(b"x")
    summary = organize(tmp_path / "dataset", tmp_path / "private", tmp_path / "fora" / "mapa.csv")
    assert summary["P01"] == {"S1": 2, "S2": 1, "nao_suportadas": 1}
    assert (tmp_path / "private" / "P01" / "S2" / "1.png").exists()
    assert (tmp_path / "dataset" / "Fulana" / "dia2" / "d.heic").exists()  # HEIC fica na origem
