import argparse

def set_parser(kind='clustering'):
    
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('mocktype', choices=['cubicbox', 'cutsky'],
                        help='The kind of mock.')
    parser.add_argument('tracer', choices=['lrg', 'elg', 'qso'],
                        help='Tracer.')
    parser.add_argument('whichmocks', choices=['firstgen', 'sv3'],
                        help='FirstGen mocks or mocks with new HOD based on sv3.')
    parser.add_argument('ph', choices=range(25), type=int, help='Phase')
    parser.add_argument('zbin', choices=range(3), type=int, nargs='?',
                        help='Redshift bin (only for CutSky)')
    parser.add_argument('-ct', '--cosmo_true', choices=['000', '003', '004'],
                        default='000')
    parser.add_argument('-cg', '--cosmo_grid', choices=[f'00{i}' for i in range(5)],
                        default='000')
    
    if kind!='recon':
        parser.add_argument('-r', '--rectype', choices=['reciso', 'recsym'],
                             help='Type of reconstruction.', default=None)
    
    return parser