"""Independent FFT evidence for source Hz, physical phase and time derivatives."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from potential import Potential


class H5FrequencyUnitsTests(unittest.TestCase):
	"""Generate source coefficients from real time samples, as the producer does."""

	def test_fft_reconstruction_and_physical_period(self) -> None:
		n, dt = 64, .037
		t = np.arange(n) * dt
		q = np.arange(8) / 8
		x, y = np.meshgrid(q, q, indexing='ij')
		mean = .2 + .03*np.cos(2*np.pi*x)
		amplitudes = np.array([.1+.02*np.sin(2*np.pi*y), .03+.01*np.cos(2*np.pi*(x+y))])
		bins = np.array([2, 6])
		phases = np.array([.37, -.61])
		f = bins/(n*dt)
		signal = mean[None] + np.sum(amplitudes[:,None]*np.cos(
			2*np.pi*f[:,None,None,None]*t[None,:,None,None]+phases[:,None,None,None]),axis=0)
		coefficients = np.fft.fft(signal,axis=0)/n
		frequencies = np.fft.fftfreq(n,d=dt)
		with tempfile.TemporaryDirectory() as folder:
			path=Path(folder)/'fft.h5'
			with h5py.File(path,'w') as h5:
				h5['Rcells']=.2+.5*q; h5['Zcells']=-.1+.5*q
				h5['freqs']=frequencies[:n//2]
				h5['fields']=coefficients[:n//2]
			p=Potential.load(path,indx=tuple(bins))
			np.testing.assert_array_equal(p.frequencies,frequencies[bins])
			np.testing.assert_array_equal(p.modes,coefficients[bins])
			actual=np.array([p.evaluate_grid(time) for time in t])
			np.testing.assert_allclose(actual,signal,rtol=1e-12,atol=1e-12)
			np.testing.assert_allclose(p.evaluate_grid(.123),p.evaluate_grid(.123+1/f[0]),rtol=1e-12,atol=1e-12)
			maximum=float(np.max(abs(actual-signal)))
			for order in (1,2):
				expected=np.sum(amplitudes[:,None]*(2*np.pi*f[:,None,None,None])**order*np.cos(
					2*np.pi*f[:,None,None,None]*t[None,:,None,None]+phases[:,None,None,None]+order*np.pi/2),axis=0)
				actual=np.array([p.evaluate_grid(time,dt=order) for time in t])
				np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
				# The spatial spline is shared; its coefficients do not define the time law.
				point=(.237,.19)
				spatial=np.array([s.evaluate(*point,dx=1) for s in p._splines[1:]])
				analytic=2*np.real(np.sum(spatial*(1j*2*np.pi*f)**order*np.exp(2j*np.pi*f*.123)))
				np.testing.assert_allclose(p.evaluate(.123,*point,dx=1,dt=order),analytic,rtol=1e-10,atol=1e-10)
			print(f'Independent FFT reconstruction max error: {maximum:.3e}')

	def test_probe_archive_conventions_and_checksum(self) -> None:
		"""Keep saved phases and verify the original archive before adaptation."""
		from dataclasses import asdict
		from hashlib import sha256
		import io
		import json
		import zipfile
		from potential import Grid
		from studies.poincare_probe import load_probe_field

		p=Potential(Grid.periodic(8,8),modes=np.ones((1,8,8),complex)*(1+.2j),
			frequencies=np.array([.37]))
		with tempfile.TemporaryDirectory() as folder:
			for convention in (None,'cycles','angular','unknown'):
				data=io.BytesIO()
				frequencies=p.frequencies*(2*np.pi if convention=='angular' else 1.)
				np.savez(data,mean=p.mean,modes=p.modes,frequencies=frequencies)
				metadata=dict(grid=asdict(p.grid),interpolation_order=3)
				if convention is not None:metadata['frequency_convention']=convention
				path=Path(folder)/'snapshot.zip'
				with zipfile.ZipFile(path,'w') as archive:
					archive.writestr('field.npz',data.getvalue())
					archive.writestr('provenance.json',json.dumps(metadata))
				digest=sha256(path.read_bytes()).hexdigest()
				if convention=='unknown':
					with self.assertRaisesRegex(ValueError,'frequency convention'):
						load_probe_field(path,expected_sha256=digest,physical_settings={})
					continue
				restored,provenance=load_probe_field(path,expected_sha256=digest,physical_settings={})
				self.assertEqual(provenance,metadata)
				for dt in (0,1,2):
					np.testing.assert_allclose(restored.evaluate_grid(.23,dt=dt),p.evaluate_grid(.23,dt=dt),rtol=1e-12,atol=1e-12)
				with self.assertRaisesRegex(ValueError,'checksum'):
					load_probe_field(path,expected_sha256='incorrect',physical_settings={})


if __name__ == '__main__':
	unittest.main()
