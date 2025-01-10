
from fastapi import FastAPI, Depends, HTTPException, Request, Form, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from datetime import datetime
from fastapi_login import LoginManager
from passlib.context import CryptContext
from sqlalchemy.ext.automap import automap_base
import logging
from fastapi import Depends, HTTPException, status, Request
from jose import jwt, JWTError
from typing import Optional
from diary_language import translate_diary
from create_quiz import make_quiz
from typing import List
from translate_quiz import translate_question,translate_quizz
from testgpt import filter_diary_entry
from wordcount import count_words
from fastapi.middleware.cors import CORSMiddleware

# Database URL
DATABASE_URL = "mysql+pymysql://root:@127.0.0.1/main"
# FastAPI app
app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
login_manager = LoginManager("your_secret_key", token_url="/login")
# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
metadata = MetaData()
metadata.reflect(bind=engine)
# Set up logging
logging.basicConfig(level=logging.INFO)
class UserPydantic(BaseModel):
    user_id: str
    name: str
    team_id: str
    password: str
    main_language: int
    learn_language: int
    icon: str
    def hash_password(self):
        self.password = pwd_context.hash(self.password)

class UserCreate(BaseModel):
    user_id: str
    name: str
    team_id: str
    password: str
    username: str
    main_language: int
    learn_language: int
    icon: str
    def hash_password(self):
        self.password = pwd_context.hash(self.password)

class TeamCreate(BaseModel):
    team_id : str
    team_name : str
    
class DiaryCreate(BaseModel):
    title: str
    content: str

class GetQuiz(BaseModel):
    user_id:str
    diary_id:int

class Multilingual_DiaryCreate(BaseModel):
    user_id :str
    language_id : int
    title : str
    diary_time :datetime
    content : str
    
class QuizCreate(BaseModel):
    diary_id : int
    question : str
    correct : str
    a : str
    b : str
    c : str
    d : str

class Multilingual_QuizCreate(BaseModel):
    diary_id : int
    language_id : int
    question : str
    correct : str
    a : str
    b : str
    c : str
    d : str

class AnswerCreate(BaseModel):
    quiz_id : int
    diary_id :int
    choices : str

class Change_User(BaseModel):
    user_name: Optional[str] = None  # 入力がある場合のみ更新
    icon: Optional[str] = None  # 入力がある場合のみ更新
    main_language: Optional[int] = None  # 入力がある場合のみ更新
    learn_language: Optional[int] = None  # 入力がある場合のみ更新
    
class Category(BaseModel):
    category1 : int
    category2 : int
    
class SelectedQuiz(BaseModel):
    selected_quizzes : List[int]

# Reflect database tables
Base = automap_base()
Base.prepare(autoload_with=engine, reflect=True)

# Table mappings
UserTable = Base.classes.user
DiaryTable = Base.classes.diary
LanguageTable = Base.classes.language
TeamTable = Base.classes.team
AnswerTable = Base.classes.answer
QuizTable = Base.classes.quiz
MQuizTable = Base.classes.multilingual_quiz
MDiaryTable = Base.classes.multilingual_diary
CashQuizTable = Base.classes.cash_quiz
# シークレットキーとアルゴリズム
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"

# トークンを解析してユーザー情報を取得
def get_current_user_from_db(token: str = Depends(oauth2_scheme)):
    try:
        # トークンをデコード
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    
    # データベースからユーザー情報を取得
    with SessionLocal() as session:
        user = session.query(UserTable).filter(UserTable.user_id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        return user  # UserTableのインスタンスを返す

@app.get("/me")
async def read_me(current_user: UserTable = Depends(get_current_user_from_db)):
    return {
        "user_id": current_user.user_id,
        "name": current_user.name,
        "team_id": current_user.team_id,
        "main_language": current_user.main_language,
    }
    
@app.get("/")
async def home():
    return JSONResponse(content={"message": "Welcome to the home page!"})

@app.post("/login")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user_id = form_data.username
    password = form_data.password
    with SessionLocal() as session:
        user = session.query(UserTable).filter(UserTable.user_id == user_id).first()
        if user and pwd_context.verify(password, user.password):
            access_token = login_manager.create_access_token(data={"sub": user_id, "main_language": user.main_language, "team_id":user.team_id})
            return {"access_token": access_token, "token_type": "bearer","status":True}
        else:
            logging.warning(f"Invalid credentials for user_id: {user_id}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

@app.post("/register")
async def user_register(user: UserCreate):
    user.hash_password()  # Hash the password
    try:
        with SessionLocal() as session:
            new_user = UserTable(
                user_id=user.user_id,
                team_id=user.team_id,
                password=user.password,
                name=user.username,
                main_language=user.main_language,
                learn_language=user.learn_language,
                icon=user.icon
            )
            session.add(new_user)
            session.commit()
        logging.info(f"User registered successfully: {user.user_id}")
        return JSONResponse({"message": "Register Successfully!"})
    except Exception as e:
        logging.error(f"Error during registration: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error during registration: {str(e)}")

@app.patch("/change_profile")
async def change_profile(
    profile_update: Change_User,  # 変更したい情報
    current_user: UserTable = Depends(get_current_user_from_db)
):
    try:
        with SessionLocal() as session:
            user_current = session.query(UserTable).filter(UserTable.user_id == current_user.user_id).first()
            if not user_current:
                raise HTTPException(status_code=404, detail="ユーザーが見つかりません")
            
            # 入力されている場合のみ更新
            if profile_update.user_name != "string":
                user_current.name = profile_update.user_name
            if profile_update.icon != "string":
                user_current.icon = profile_update.icon
            if profile_update.main_language != 0:
                user_current.main_language = profile_update.main_language
            if profile_update.learn_language != 0:
                # 学習言語の存在確認
                language_exists = session.query(LanguageTable).filter(LanguageTable.language_id == profile_update.learn_language).first()
                if not language_exists:
                    raise HTTPException(status_code=400, detail="指定された学習言語は存在しません")
                user_current.learn_language = profile_update.learn_language

            session.commit()

    except Exception as e:
        raise HTTPException(status_code=500, detail="プロフィール更新中にエラーが発生しました")
    
    return {"message": "プロフィールが正常に更新されました！"}

@app.post('/team_register')
async def team_register(team: TeamCreate):
    try:
        with SessionLocal() as session:
            new_team = TeamTable(
                team_id = team.team_id,
                team_name = team.team_name,
                team_time = datetime.now()
            )
            session.add(new_team)
            session.commit()
            logging.info(f"User registered successfully: {team.team_id}")
        return JSONResponse({"message": "Register Successfully!"})
    except Exception as e:
        logging.error(f"Error during registration: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error during registration: {str(e)}")


@app.post("/generate_quiz")
async def generate_quiz(category: Category, current_user: UserTable = Depends(get_current_user_from_db)):
    try:
        with SessionLocal() as session:
            # 最後の日記を取得
            result = (session.query(MDiaryTable)
                      .filter(MDiaryTable.language_id == current_user.main_language)
                      .order_by(MDiaryTable.diary_time.desc())
                      .first())
            
            # 日記が存在しない場合の処理
            if result is None:
                return JSONResponse(status_code=404, content={"error": "No diary found."})

            # クイズを生成
            quizzes = make_quiz(result.content, category.category1, category.category2)

            # クイズが生成されなかった場合の処理
            if not quizzes:
                return JSONResponse(status_code=404, content={"error": "No quizzes generated."})
            # 既存のキャッシュを削除（同じユーザーの古いキャッシュがある場合）
            session.query(CashQuizTable).filter(CashQuizTable.user_id == current_user.user_id).delete()
            session.commit()
            # キャッシュテーブルに保存
            for  i, quiz_data in enumerate(quizzes):
                new_cache = CashQuizTable(
                    cash_quiz_id = i + 1,
                    diary_id=result.diary_id,
                    user_id=current_user.user_id,
                    question=quiz_data['question'],
                    correct=quiz_data['answer'].split(":")[1].strip(),
                    a=quiz_data['choices'][0].split(".")[1].strip(),
                    b=quiz_data['choices'][1].split(".")[1].strip(),
                    c=quiz_data['choices'][2].split(".")[1].strip(),
                    d=quiz_data['choices'][3].split(".")[1].strip(),
                )
                session.add(new_cache)
            session.commit()
            
            if current_user.main_language == 1:
                translated_questions = []
                for quiz_data in quizzes:
                    translated_questions.append(quiz_data["question"])
                # JSON形式でクイズを返す
                return JSONResponse(content={"quizzes": translated_questions})
            else:
                # translate_quiz() に引数を渡して翻訳を実行
                translated_questions = []
                for quiz_data in quizzes:
                    translated_question = translate_question(quiz_data['question'],current_user.main_language)
                    translated_questions.append(translated_question)
                return JSONResponse(content={"quizzes": translated_questions})
    except Exception as e:
        # エラーが発生した場合は500エラーを返す
        return JSONResponse(status_code=500, content={"error": f"An error occurred: {str(e)}"})
    
@app.post("/save_quiz")
async def save_quiz(selected_quizzes: SelectedQuiz, current_user: UserTable = Depends(get_current_user_from_db)):
    try:
        with SessionLocal() as session:
            # クイズ情報が5問選ばれているかを確認
            if len(selected_quizzes.selected_quizzes) != 5:
                return JSONResponse(status_code=400, content={"error": "You must select exactly 5 questions."})

            # キャッシュテーブルから選ばれたクイズ情報を取得
            selected_quiz_ids = selected_quizzes.selected_quizzes  # リストのまま使用
            quizzes_to_save = session.query(CashQuizTable).filter(
                CashQuizTable.cash_quiz_id.in_(selected_quiz_ids),
                CashQuizTable.user_id == current_user.user_id
            ).all()

            if len(quizzes_to_save) != 5:
                return JSONResponse(status_code=404, content={"error": "Selected quizzes not found in cache."})

            # クイズ情報を正式なテーブルに保存
            for i,quiz in enumerate(quizzes_to_save):  
                new_quiz = QuizTable(
                    quiz_id= i + 1  ,
                    diary_id=quiz.diary_id,  
                    question=quiz.question,  
                    correct=quiz.correct,
                    a=quiz.a,
                    b=quiz.b,
                    c=quiz.c,
                    d=quiz.d
                    )
                session.add(new_quiz)
                session.flush()  # new_quiz.quiz_idを取得するためにフラッシュ

                translated_quizzes_to_save = translate_quizz(quiz.quiz,quiz.a,quiz.b,quiz.c,quiz.d)
                for i, translated_quiz in enumerate(translated_quizzes_to_save):
                    new_translate_quiz = MQuizTable(
                        quiz_id=new_quiz.quiz_id,
                        diary_id=quiz.diary_id,
                        language_id=i + 1,
                        question=translated_quiz.question,
                        correct=quiz.correct,
                        a=translated_quiz.a,
                        b=translated_quiz.b,
                        c=translated_quiz.c,
                        d=translated_quiz.d
                    )
                    session.add(new_translate_quiz)

            # キャッシュをクリア
            session.query(CashQuizTable).filter(CashQuizTable.user_id == current_user.user_id).delete()
            session.commit()

            logging.info("Successfully saved selected quizzes.")
            return JSONResponse(content={"message": "Selected quizzes saved successfully."})
    except Exception as e:
        logging.error(f"Error saving selected quizzes: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"An error occurred: {str(e)}"})
@app.post("/add_diary")
async def add_diary(diary: DiaryCreate, current_user: UserTable = Depends(get_current_user_from_db)):
    """
    現在ログインしているユーザーの情報を利用して日記を追加します。
    """
    diary_time = datetime.now()  # 現在時刻を取得
    
    complaining = filter_diary_entry(diary.content)
    wordcount = count_words(diary.content,current_user.main_language)
    if complaining == 1 or complaining == 2 or wordcount <=200:
        return{"status":True,"message":"悪口が含まれている可能性があるか、文字数が200文字に達していません。書き直してください。"}
    else:
        with SessionLocal() as session:
                try:
                    # DiaryTableに新しいエントリを追加
                    new_diary = DiaryTable(
                        user_id=current_user.user_id,
                        title=diary.title,  # diary.titleを使用
                        diary_time=diary_time,
                        content=diary.content,  # diary.contentを使用
                        main_language=current_user.main_language
                    )
                    session.add(new_diary)
                    session.commit()  # データを確定
                    session.refresh(new_diary)  # 新しいエントリのIDを取得可能にする

                    # 翻訳された日記を追加
                    diary_id = new_diary.diary_id
                    diary_list = translate_diary(diary.title, diary.content,current_user.main_language)
                    
                    translated_entries = []
                    for i, (title, content) in enumerate(diary_list, start=1):
                        translated_entries.append(MDiaryTable(
                            diary_id=diary_id,
                            language_id=i,
                            user_id=current_user.user_id,
                            title=title,
                            diary_time=diary_time,
                            content=content,
                        ))
                    
                    session.add_all(translated_entries)  # 複数のエントリを一括追加
                    session.commit()  # データを確定
                    
                    logging.info(
                        f"Diary added successfully: user_id={current_user.user_id}, diary_id={diary_id}"
                    )
                except Exception as e:
                    session.rollback()  # エラー時にロールバック
                    logging.error(f"Error while adding diary: {e}")
                    raise e
        return {"status":False,"message": "Diary added successfully!"}



@app.get("/get_diaries")
async def get_diaries(current_user: UserTable = Depends(get_current_user_from_db)):
    """
    チームに所属する全てのユーザーの日記を取得し、
    現在ログインしているユーザーのmain_languageで出力します。
    """
    team_id = current_user.team_id  # 現在のユーザーの team_id を取得
    main_language = current_user.main_language  # 現在のユーザーの main_language を取得
    
    
    with SessionLocal() as session:
        # multilingual_diaryテーブルからチームに所属するユーザーの日記を取得し、翻訳情報を結合
        result = (
            session.query(
                MDiaryTable.user_id,
                MDiaryTable.diary_id,
                MDiaryTable.title,
                MDiaryTable.content,
                MDiaryTable.diary_time,
            )
            .join(DiaryTable, DiaryTable.diary_id == MDiaryTable.diary_id)  # DiaryTableと結合
            .join(UserTable, UserTable.user_id == MDiaryTable.user_id)  # UserTableと結合
            .filter(UserTable.team_id == team_id)  # チームIDでフィルタ
            .filter(MDiaryTable.language_id == main_language)  # main_languageでフィルタ
            .order_by(DiaryTable.diary_time.desc())  # 日記の時間で並び替え
            .all()
        )

    # 結果を整形
    diaries = [
        {
            "user_id": row.user_id,
            "diary_id": row.diary_id,
            "title": row.title,
            "content": row.content,
            "diary_time": row.diary_time.strftime('%Y-%m-%d %H:%M:%S')
        }
        for row in result
    ]

    return JSONResponse(content={"team_id": team_id, "diaries": diaries})

@app.get("/get_same_quiz")
async def get_same_quiz(quizzes: GetQuiz, current_user: UserTable = Depends(get_current_user_from_db)):
    try:
        with SessionLocal() as session:
            # 指定された日記IDに基づいて、関連する全てのクイズを取得
            quiz_results = (
                session.query(MQuizTable)
                .filter(MQuizTable.diary_id == quizzes.diary_id)
                .all()
            )

            quizzes_data = []
            for q in quiz_results:
                # 問題文はユーザーの母国語で取得
                if q.language_id == current_user.main_language:
                    question = q.question  # 自分の言語の問題文
                    choices = {
                        "a": q.a,
                        "b": q.b,
                        "c": q.c,
                        "d": q.d
                    }
                else:
                    continue  # 母国語以外は無視する
                
                # クイズデータをリストに追加
                quizzes_data.append({
                        "quiz_id": q.quiz_id,
                        "question": question,
                        "choices": choices
                    })

            return JSONResponse(content={"quizzes": quizzes_data})
    except Exception as e:
        logging.error(f"Error during getting quiz: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error during getting quiz: {str(e)}")
  

@app.get("get_different_quiz")
async def get_different_quiz(quizzes: GetQuiz, current_user: UserTable = Depends(get_current_user_from_db)):
    try:
        with SessionLocal() as session:
            # 指定された日記IDに基づいて、関連する全てのクイズを取得
            quiz_results = (
                session.query(MQuizTable)
                .filter(MQuizTable.diary_id == quizzes.diary_id)
                .all()
            )

            quizzes_data = []
            for q in quiz_results:
                # 問題文はユーザーの母国語で取得
                if q.language_id == current_user.main_language:
                    question = q.question  # 自分の言語の問題文
                else:
                    continue  # 母国語以外は無視する

                # 選択肢は習得したい言語で取得
                choices_query = (
                    session.query(MQuizTable)
                    .filter(MQuizTable.quiz_id == q.quiz_id, MQuizTable.language_id == current_user.learn_language)
                    .first()
                )

                # 選択肢が取得できた場合のみ追加
                if choices_query:
                    choices = {
                        "a": choices_query.a,
                        "b": choices_query.b,
                        "c": choices_query.c,
                        "d": choices_query.d
                    }

                    # クイズデータをリストに追加
                    quizzes_data.append({
                        "quiz_id": q.quiz_id,
                        "question": question,
                        "choices": choices
                    })

            return JSONResponse(content={"quizzes": quizzes_data})
    except Exception as e:
        logging.error(f"Error during getting quiz: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error during getting quiz: {str(e)}")
    
    
@app.post("/create_answer")
async def create_answer(answer : AnswerCreate, current_user: UserTable = Depends(get_current_user_from_db)):
    try:
        answer_time = datetime.now()  # 現在時刻を取得
        with SessionLocal() as session:
            quiz = session.query(QuizTable).filter(QuizTable.diary_id == answer.diary_id,QuizTable.quiz_id == answer.quiz_id).first()
            if not quiz:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Quiz with id {answer.quiz_id} not found."
                )
            if quiz.correct == answer.choices:
                judgement = 1
            else:
                judgement = 0
                
            new_answer = AnswerTable(
                user_id=current_user.user_id,
                quiz_id=answer.quiz_id,
                diary_id=answer.diary_id,
                language_id=current_user.main_language,
                answer_date=answer_time,
                choices=answer.choices,
                judgement=judgement
            )
            session.add(new_answer)
            session.commit()
            logging.info(f"Answer created successfully for user_id: {current_user.user_id}")
        return JSONResponse({"message": "Answer Created Successfully!"})

    except Exception as e:
        logging.error(f"Error during creating answer: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error during creating answer: {str(e)}")

@app.get("/get_answer")
async def get_answer(current_user: UserTable = Depends(get_current_user_from_db)):
    try:
        with SessionLocal() as session:
                results = session.query(AnswerTable).filter(AnswerTable.user_id == current_user.user_id).order_by(AnswerTable.answer_date.desc()).all()
        answers = [
            {
                "user_id": row.user_id,
                "diary_id": row.diary_id,
                "language_id": row.language_id,
                "answer_date": row.answer_date.strftime('%Y-%m-%d %H:%M:%S'),
                "choices": row.choices,
                "judgement": row.judgement,
            }
            for row in results
        ]
        return JSONResponse(content={"answers": answers})
    
    except Exception as e:
        logging.error(f"Error during creating answer: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error during creating answer: {str(e)}")

@app.exception_handler(404)
async def page_not_found(request: Request, exc):
    return JSONResponse(status_code=404, content={"error": "Page not found"})

@app.exception_handler(500)
async def internal_server_error(request: Request, exc):
    return JSONResponse(status_code=500, content={"error": "Internal server error"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")