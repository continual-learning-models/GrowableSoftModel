"""105 S-2: base model-policy key gate (design 104 v1.3,
FR-1 name gate; FR-5 deferred). T-1..T-6 boxes; written RED
before implementation. T-5's SMS legs live in
SoftModelSystem/tests (surface parity via the same refusal
text) and are exercised here through subprocess probes into
the SMS venv when it is present."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

LIB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LIB))
sys.path.insert(0, str(LIB / "modules" / "Generator"))
import core._modules  # noqa: F401,E402  (module path wiring)
from core.facade import System  # noqa: E402
from core.lifecycle import DEFAULT_POLICY  # noqa: E402

SMS = LIB.parent / "SoftModelSystem"
SMS_PY = SMS / ".venv" / "bin" / "python"


def _rows(n, d=3):
    return [{"input": {f"f{j}": float(i + j) for j in range(d)},
             "target": float(i)} for i in range(n)]


class _Box(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["SOFTMODEL_MODELS_ROOT"] = self._tmp.name
        self.s = System()

    def tearDown(self):
        os.environ.pop("SOFTMODEL_MODELS_ROOT", None)
        self._tmp.cleanup()

    def _mk(self, mid="m"):
        out = self.s.create_model(mid, holdout=_rows(4))
        self.assertNotIn("refusal", out)
        return mid

    def _pol_file(self, mid):
        return self.s.lc._mdir(mid) / "policy.json"

    def _pol_hash(self, mid):
        return hashlib.sha256(
            self._pol_file(mid).read_bytes()).hexdigest()


class T1LifeDoorNameGate(_Box):
    def test_t1(self):
        m = self._mk()
        before = self._pol_hash(m)
        out = self.s.set_policy(m, gate_toll=0.4)
        self.assertIn("refusal", out, out)
        self.assertIn("gate_toll", out["refusal"])
        self.assertEqual(before, self._pol_hash(m),
                         "policy.json must be byte-unchanged")
        served = self.s.infer(m, _rows(1)[0]["input"])
        self.assertIn("output", served)


class T2MultiOffender(_Box):
    def test_t2(self):
        m = self._mk()
        before = self._pol_hash(m)
        out = self.s.set_policy(m, zzz=2, aaa=1)
        self.assertIn("refusal", out, out)
        self.assertIn("['aaa', 'zzz']", out["refusal"],
                      "all offenders, sorted, in one refusal")
        self.assertEqual(before, self._pol_hash(m))


class T3BirthDoorGate(_Box):
    def test_t3(self):
        out = self.s.create_model("t3", holdout=_rows(4),
                                  policy={"typo": 1})
        self.assertIn("refusal", out, out)
        self.assertIn("typo", out["refusal"])
        t3_dirs = [p for p in Path(self._tmp.name).rglob("t3")
                   if p.is_dir()]
        self.assertEqual(t3_dirs, [],
                         "refusal must leave no model directory")
        listed = self.s.list_models()
        ids = [r.get("model_id") for r in
               (listed.get("models", listed)
                if isinstance(listed, dict) else listed)]
        self.assertNotIn("t3", ids)
        leftovers = [p for p in
                     Path(self._tmp.name).rglob("*t3*")]
        self.assertEqual(leftovers, [],
                         "no holdout store or any state for t3")


class T4ValidFamiliesStillPass(_Box):
    def test_t4(self):
        m = self._mk()
        for kw in ({"max_params_mult": 12},
                   {"gate_recent_n": None},
                   {"spu_enabled": True},
                   {"growth_params": {"stall_k": 4,
                                      "rl.horizon": 8,
                                      "preference.rule":
                                          "thompson"}}):
            out = self.s.set_policy(m, **kw)
            self.assertNotIn("refusal", out, (kw, out))
        pol = json.loads(self._pol_file(m).read_text())
        self.assertEqual(pol["max_params_mult"], 12)
        self.assertIsNone(pol["gate_recent_n"])
        self.assertTrue(pol["spu_enabled"])
        self.assertEqual(pol["growth_params"]["stall_k"], 4)
        self.assertEqual(pol["growth_params"]["rl.horizon"], 8)
        self.assertEqual(
            pol["growth_params"]["preference.rule"], "thompson")

    def test_t4_att_on_ga_substrate(self):
        out = self.s.create_model(
            "ga", holdout=_rows(6, 4),
            substrate="growable_attention")
        self.assertNotIn("refusal", out)
        out = self.s.set_policy("ga", att_lambda=0.001)
        self.assertNotIn("refusal", out, out)
        pol = json.loads(self._pol_file("ga").read_text())
        self.assertEqual(pol["att_lambda"], 0.001)


class T5SurfaceParity(_Box):
    BAD = "gate_toll"

    def _expect(self):
        m = self._mk()
        out = self.s.set_policy(m, **{self.BAD: 0.4})
        self.assertIn("refusal", out)
        return out["refusal"]

    def test_t5a_facade(self):
        self.assertIn(self.BAD, self._expect())

    def test_t5b_cli(self):
        text = self._expect()
        args = json.dumps({"model_id": "m", self.BAD: 0.4})
        r = subprocess.run(
            [sys.executable, "-m", "cli.cli", "set_policy",
             args],
            capture_output=True, text=True, cwd=str(LIB),
            env={**os.environ,
                 "SOFTMODEL_MODELS_ROOT": self._tmp.name})
        self.assertIn(text, r.stdout,
                      (r.stdout, r.stderr))

    def test_t5c_mcp_inprocess(self):
        text = self._expect()
        out = self.s.set_policy("m", **{self.BAD: 0.4})
        self.assertEqual(out["refusal"], text,
                         "MCP dispatches s.set_policy(**updates)"
                         " — same call, same text")

    @unittest.skipUnless(SMS_PY.exists(), "SMS venv absent")
    def test_t5defg_sms_surfaces(self):
        # Review F8 / T-20: the legs are only meaningful if the
        # SMS venv resolves the lib to THIS working tree
        # (editable install) — assert the premise explicitly.
        r = subprocess.run(
            [str(SMS_PY), "-c",
             "import core; print(core.__file__)"],
            capture_output=True, text=True, cwd=str(SMS))
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        self.assertIn(str(LIB), r.stdout,
                      "SMS venv must editable-resolve the lib "
                      "to this tree; got: " + r.stdout)
        text = self._expect()
        probe = (
            "import json,sys,tempfile\n"
            "sys.path.insert(0,'.')\n"
            "from sms.operator.api import Operator\n"
            "from sms.common.errors import Refusal\n"
            "ws=tempfile.mkdtemp();op=Operator(ws)\n"
            "op.schema_declare([{'name':'f0'},{'name':'f1'}],"
            "'y',target_kind='numeric')\n"
            "rows=[{'f0':float(i),'f1':float(i+1),"
            "'y':float(i)} for i in range(8)]\n"
            "rec=op.ingest_rows(rows)\n"
            "man=op.snapshot_build('s',[rec['id']])\n"
            "snap='s-'+man['hash']\n"
            "msgs=[]\n"
            "try:\n"
            "  op.train_converge('mm',snap,patience=3,"
            "steps_per_batch=20,policy={'gate_toll':0.4})\n"
            "except Refusal as e: msgs.append(('e',str(e)))\n"
            "try:\n"
            "  op.train_converge('mm',snap,patience=3,"
            "steps_per_batch=20)\n"
            "except Refusal as e: pass\n"
            "try:\n"
            "  op.model_policy('mm',{'gate_toll':0.4})\n"
            "except Refusal as e: msgs.append(('d',str(e)))\n"
            "from sms.operator.mcp_server import SMSServer\n"
            "srv=SMSServer(ws)\n"
            "assert any(t['name']=='model_policy'"
            " for t in srv.tools)\n"
            "resp=srv.handle({'id':1,'jsonrpc':'2.0',"
            "'method':'tools/call','params':{'name':"
            "'model_policy','arguments':{'model_id':'mm',"
            "'updates':{'gate_toll':0.4}}}})\n"
            "body=json.loads(resp['result']['content'][0]"
            "['text']) if 'result' in resp else resp\n"
            "msgs.append(('g',json.dumps(body)))\n"
            "print(json.dumps(msgs))\n")
        r = subprocess.run([str(SMS_PY), "-c", probe],
                           capture_output=True, text=True,
                           cwd=str(SMS))
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        msgs = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertGreaterEqual(len(msgs), 3, msgs)
        for tag, msg in msgs:
            self.assertIn(text, msg, (tag, msg))
        # (f) SMS CLI dispatcher
        cli = subprocess.run(
            [str(SMS_PY), "-m", "sms.operator.cli",
             tempfile.mkdtemp(), "model_policy",
             json.dumps({"model_id": "nope",
                         "updates": {self.BAD: 1}})],
            capture_output=True, text=True, cwd=str(SMS))
        self.assertTrue(self.BAD in cli.stdout
                        or self.BAD in cli.stderr,
                        (cli.stdout, cli.stderr))


class T6WhitelistIntegrity(_Box):
    def test_t6(self):
        from core.lifecycle import (VALID_POLICY_KEYS,
                                    VALID_POLICY_PREFIXES)
        self.assertEqual(VALID_POLICY_KEYS,
                         set(DEFAULT_POLICY))
        self.assertEqual(VALID_POLICY_PREFIXES,
                         ("spu_", "att_"))



class T8PrefixedTypoAtBirth(_Box):
    """104 v1.4: a prefixed typo must refuse at the BIRTH door
    too (the S-6.2 hole: name gate admits the prefix, so the
    family validator must rule there as well)."""

    def test_t8_spu(self):
        out = self.s.create_model("t8", holdout=_rows(4),
                                  policy={"spu_typo": 1})
        self.assertIn("refusal", out, out)
        self.assertIn("spu_typo", out["refusal"])
        leftovers = [p for p in
                     Path(self._tmp.name).rglob("*t8*")]
        self.assertEqual(leftovers, [], "zero state on refusal")

    def test_t8_spu_bad_value(self):
        out = self.s.create_model("t8v", holdout=_rows(4),
                                  policy={"spu_enabled":
                                          "banana"})
        self.assertIn("refusal", out, out)

    def test_t8_att(self):
        out = self.s.create_model("t8a", holdout=_rows(4),
                                  policy={"att_typo": 1})
        self.assertIn("refusal", out, out)
        self.assertIn("att_typo", out["refusal"])


class T9InnovationValuesAtLifeDoor(_Box):
    """104 v1.4: set_policy runs the same innovation_* checks
    create_model already has (keys vs V27_DEFAULTS + method
    enum)."""

    def test_t9_bad_method(self):
        m = self._mk()
        before = self._pol_hash(m)
        out = self.s.set_policy(m, innovation_method="bogus")
        self.assertIn("refusal", out, out)
        self.assertIn("bogus", out["refusal"])
        self.assertEqual(before, self._pol_hash(m))

    def test_t9_valid_method_passes(self):
        m = self._mk()
        out = self.s.set_policy(m, innovation_progress_eps=0.02)
        self.assertNotIn("refusal", out, out)


class T10SubstrateParamsAtLifeDoor(_Box):
    """104 v1.4 addendum: substrate_params inner keys get the
    same signature-filtered check at the life door that the
    birth door has (last family without door symmetry — the
    round-2 close-out finding)."""

    def test_t10_typo_refused(self):
        m = self._mk()
        before = self._pol_hash(m)
        out = self.s.set_policy(
            m, substrate_params={"lr_typo": 0.1})
        self.assertIn("refusal", out, out)
        self.assertIn("lr_typo", out["refusal"])
        self.assertEqual(before, self._pol_hash(m))

    def test_t10_valid_inner_passes(self):
        m = self._mk()
        out = self.s.set_policy(m, substrate_params={"lr": 0.05})
        self.assertNotIn("refusal", out, out)

class T11SubstrateValueBothDoors(_Box):
    """Review F1: the substrate VALUE is registry-checked even
    when it arrives inside the policy dict, at both doors."""

    def test_t11_birth_typo(self):
        out = self.s.create_model(
            "t11", holdout=_rows(4),
            policy={"substrate": "growable_attn"})
        self.assertIn("refusal", out, out)
        self.assertIn("growable_attn", out["refusal"])
        leftovers = [p for p in
                     Path(self._tmp.name).rglob("*t11*")]
        self.assertEqual(leftovers, [], "zero state on refusal")

    def test_t11_life_typo(self):
        m = self._mk()
        before = self._pol_hash(m)
        out = self.s.set_policy(m, substrate="growable_attn")
        self.assertIn("refusal", out, out)
        self.assertEqual(before, self._pol_hash(m))


class T12SpuTypeErrorIsRefusal(_Box):
    """Review F2: non-numeric spu values refuse (no exception
    escapes), both doors."""

    def test_t12_birth(self):
        out = self.s.create_model("t12", holdout=_rows(4),
                                  policy={"spu_eta": "fast"})
        self.assertIn("refusal", out, out)

    def test_t12_life_none(self):
        m = self._mk()
        out = self.s.set_policy(m, spu_p_mask=None)
        self.assertIn("refusal", out, out)


class T13UnknownStoredSubstrate(_Box):
    """Review F3: life-door substrate_params check refuses
    coherently when the stored substrate is not registered."""

    def test_t13(self):
        m = self._mk()
        f = self._pol_file(m)
        pol = json.loads(f.read_text())
        pol["substrate"] = "ghost_sub"
        f.write_text(json.dumps(pol))
        out = self.s.set_policy(m, substrate_params={"lr": 0.05})
        self.assertIn("refusal", out, out)
        self.assertIn("ghost_sub", out["refusal"])
        self.assertNotIn("'args'", out["refusal"],
                         "no signature-of-None nonsense")


class T14SameCallSubstrateSwitch(_Box):
    """Review F4 (regression): substrate + substrate_params for
    the NEW substrate in one set_policy call is accepted again
    (validated against the same-call substrate)."""

    def test_t14(self):
        m = self._mk()
        out = self.s.set_policy(m, substrate="transformer",
                                substrate_params={"backend": 2})
        self.assertNotIn("refusal", out, out)
        pol = json.loads(self._pol_file(m).read_text())
        self.assertEqual(pol["substrate"], "transformer")
        self.assertEqual(pol["substrate_params"]["backend"], 2)

    def test_t14_bad_params_for_new_substrate(self):
        m = self._mk()
        out = self.s.set_policy(m, substrate="transformer",
                                substrate_params={"lr_typo": 1})
        self.assertIn("refusal", out, out)
        self.assertIn("transformer", out["refusal"])


class T15SpuIncrementalVsStored(_Box):
    """Review F5: spu cross-key constraints validate against the
    model's STORED spu policy, not the defaults."""

    def test_t15_valid_incremental(self):
        out = self.s.create_model(
            "t15", holdout=_rows(4),
            policy={"spu_newborn_steps": 1000})
        self.assertNotIn("refusal", out)
        out = self.s.set_policy("t15", spu_warmup_steps=500)
        self.assertNotIn("refusal", out, out)

    def test_t15_genuinely_bad_still_refused(self):
        self.s.create_model("t15b", holdout=_rows(4),
                            policy={"spu_newborn_steps": 1000})
        out = self.s.set_policy("t15b", spu_warmup_steps=2000)
        self.assertIn("refusal", out, out)


class T16InnovationValuesBothDoors(_Box):
    """Review F6: innovation_* VALUES validate at both doors
    (full _validate_innovation, not just the method enum)."""

    def test_t16_life(self):
        m = self._mk()
        before = self._pol_hash(m)
        out = self.s.set_policy(m, innovation_progress_eps=5)
        self.assertIn("refusal", out, out)
        self.assertIn("innovation_progress_eps", out["refusal"])
        self.assertEqual(before, self._pol_hash(m))

    def test_t16_birth(self):
        out = self.s.create_model(
            "t16", holdout=_rows(4),
            policy={"innovation_progress_eps": 5})
        self.assertIn("refusal", out, out)


class T17GhostModelRefusal(_Box):
    """Review F7: set_policy on a nonexistent model refuses
    (outcome, not exception) and creates nothing."""

    def test_t17(self):
        out = self.s.set_policy("ghost", max_params_mult=5)
        self.assertIn("refusal", out, out)
        self.assertIn("ghost", out["refusal"])
        self.assertFalse(
            (Path(self._tmp.name) / "ghost").exists())


class T18NonStringKey(_Box):
    """Review F9: a non-string policy key refuses instead of
    crashing the gate."""

    def test_t18(self):
        out = self.s.create_model("t18", holdout=_rows(4),
                                  policy={1: 2})
        self.assertIn("refusal", out, out)


class T19NonDictSubstrateParams(_Box):
    """Review F10: a non-dict substrate_params refuses with a
    type message, not a character-soup key list."""

    def test_t19_life(self):
        m = self._mk()
        out = self.s.set_policy(m, substrate_params="lr")
        self.assertIn("refusal", out, out)
        self.assertNotIn("'l'", out["refusal"])

    def test_t19_birth(self):
        out = self.s.create_model("t19", holdout=_rows(4),
                                  policy={"substrate_params":
                                          ["lr"]})
        self.assertIn("refusal", out, out)


if __name__ == "__main__":
    unittest.main()
